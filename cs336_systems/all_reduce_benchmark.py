from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import statistics
import time

import torch
import torch.distributed as dist


# 使用 SI 单位：
# 1 MB = 1,000,000 bytes
# 1 GB = 1,000,000,000 bytes
MESSAGE_SIZES = {
    "1MB": 1_000_000,
    "10MB": 10_000_000,
    "100MB": 100_000_000,
    "1GB": 1_000_000_000,
}

CSV_FIELDS = [
    "hostname",
    "gpu_name",
    "backend",
    "world_size",
    "message_size",
    "message_bytes",
    "num_elements",
    "dtype",
    "warmup_steps",
    "measurement_steps",
    "mean_ms",
    "std_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "algorithmic_bandwidth_GBps",
    "bus_bandwidth_GBps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=MESSAGE_SIZES,
        default=list(MESSAGE_SIZES),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--measurement-steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "profiles/all_reduce_benchmark.csv"
        ),
    )

    return parser.parse_args()


def append_csv_row(
    output_path: Path,
    row: dict[str, object],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    with output_path.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def benchmark_all_reduce(
    tensor: torch.Tensor,
    warmup_steps: int,
    measurement_steps: int,
) -> list[float]:
    # NCCL 初始化和算法选择都有首次调用开销。
    for _ in range(warmup_steps):
        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
            async_op=False,
        )
        torch.cuda.synchronize()

    timings_ms: list[float] = []

    for _ in range(measurement_steps):
        # 确保所有 rank 同时开始本轮实验。
        dist.barrier()
        torch.cuda.synchronize()

        start_ns = time.perf_counter_ns()

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
            async_op=False,
        )

        # async_op=False 不等于 GPU 已经执行完毕，
        # 因此这里仍然必须 synchronize。
        torch.cuda.synchronize()

        end_ns = time.perf_counter_ns()

        timings_ms.append(
            (end_ns - start_ns) / 1_000_000
        )

    return timings_ms


def summarize_timings(
    timings_from_all_ranks: list[list[float]],
    message_bytes: int,
    world_size: int,
) -> dict[str, float]:
    measurement_steps = len(
        timings_from_all_ranks[0]
    )

    # 一次 collective 的整体完成时间由最慢 rank 决定。
    per_iteration_max_ms = [
        max(
            rank_timings[step]
            for rank_timings in timings_from_all_ranks
        )
        for step in range(measurement_steps)
    ]

    mean_ms = statistics.fmean(
        per_iteration_max_ms
    )

    std_ms = (
        statistics.stdev(per_iteration_max_ms)
        if len(per_iteration_max_ms) >= 2
        else 0.0
    )

    # Algorithmic bandwidth = message size / latency
    algorithmic_bandwidth_GBps = (
        message_bytes / 1_000_000_000
    ) / (mean_ms / 1000)

    # Ring all-reduce 常用的 bus bandwidth 归一化。
    bus_bandwidth_GBps = (
        algorithmic_bandwidth_GBps
        * 2
        * (world_size - 1)
        / world_size
    )

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": statistics.median(
            per_iteration_max_ms
        ),
        "min_ms": min(per_iteration_max_ms),
        "max_ms": max(per_iteration_max_ms),
        "algorithmic_bandwidth_GBps": (
            algorithmic_bandwidth_GBps
        ),
        "bus_bandwidth_GBps": (
            bus_bandwidth_GBps
        ),
    }


def main() -> None:
    args = parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size > 6:
        raise ValueError(
            "The assignment permits at most 6 GPUs."
        )

    if world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {world_size} processes, but only "
            f"{torch.cuda.device_count()} GPUs are visible."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(
        "cuda",
        local_rank,
    )

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=5),
    )

    try:
        for size_name in args.sizes:
            message_bytes = MESSAGE_SIZES[size_name]

            # float32 每个元素为 4 bytes。
            num_elements = message_bytes // 4

            # 使用全零 tensor，反复 SUM 后仍然为零，
            # 避免数值不断放大导致溢出。
            tensor = torch.zeros(
                num_elements,
                device=device,
                dtype=torch.float32,
            )

            # 确保所有 rank 都完成 tensor 分配。
            dist.barrier()

            local_timings = benchmark_all_reduce(
                tensor=tensor,
                warmup_steps=args.warmup_steps,
                measurement_steps=(
                    args.measurement_steps
                ),
            )

            # 收集每个 rank 的测量结果。
            timings_from_all_ranks = [
                None
                for _ in range(world_size)
            ]

            dist.all_gather_object(
                timings_from_all_ranks,
                local_timings,
            )

            if rank == 0:
                summary = summarize_timings(
                    timings_from_all_ranks,
                    message_bytes=message_bytes,
                    world_size=world_size,
                )

                row = {
                    "hostname": socket.gethostname(),
                    "gpu_name": (
                        torch.cuda.get_device_name(
                            local_rank
                        )
                    ),
                    "backend": "nccl",
                    "world_size": world_size,
                    "message_size": size_name,
                    "message_bytes": message_bytes,
                    "num_elements": num_elements,
                    "dtype": "float32",
                    "warmup_steps": (
                        args.warmup_steps
                    ),
                    "measurement_steps": (
                        args.measurement_steps
                    ),
                    **summary,
                }

                append_csv_row(
                    args.output,
                    row,
                )

                print(
                    json.dumps(
                        row,
                        indent=2,
                    )
                )

            del tensor
            torch.cuda.empty_cache()

            # 等待 rank 0 保存完结果。
            dist.barrier()

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()