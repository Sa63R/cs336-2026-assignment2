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

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp import FlatDDP, NaiveDDP


MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

CSV_FIELDS = [
    "hostname",
    "gpu_name",
    "world_size",
    "strategy",
    "model_size",
    "dtype",
    "global_batch_size",
    "local_batch_size",
    "context_length",
    "num_parameters",
    "gradient_size_gib",
    "all_reduce_calls",
    "warmup_steps",
    "measurement_steps",
    "iteration_mean_ms",
    "iteration_std_ms",
    "gradient_sync_mean_ms",
    "gradient_sync_std_ms",
    "communication_percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["naive", "flat"],
        required=True,
    )
    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS,
        default="xl",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default="float32",
    )
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/ddp_flat_benchmark.csv"),
    )
    return parser.parse_args()


def append_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def synchronize() -> None:
    torch.cuda.synchronize()


def aggregate_rank_timings(
    local_timings: list[float],
    world_size: int,
) -> list[float]:
    gathered: list[list[float] | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(gathered, local_timings)

    # 一次 distributed step 的完成时间由最慢的 rank 决定。
    return [
        max(
            rank_timings[step]
            for rank_timings in gathered
            if rank_timings is not None
        )
        for step in range(len(local_timings))
    ]


def mean_and_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, std


def main() -> None:
    args = parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if args.global_batch_size % world_size != 0:
        raise ValueError(
            "--global-batch-size must be divisible by WORLD_SIZE"
        )
    if world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=30),
        device_id=device,
    )

    try:
        config = MODEL_CONFIGS[args.model_size]
        dtype = DTYPES[args.dtype]
        local_batch_size = args.global_batch_size // world_size

        # 让各 rank 初始权重不同，以验证 DDP 构造时的 rank-0 broadcast。
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)

        with torch.device(device):
            base_model = BasicsTransformerLM(
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                d_model=config["d_model"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                d_ff=config["d_ff"],
            )
        base_model = base_model.to(dtype=dtype)

        model = (
            NaiveDDP(base_model)
            if args.strategy == "naive"
            else FlatDDP(base_model)
        )
        model.train()

        # 使用无状态的 SGD，避免 Adam 状态和 optimizer 计算掩盖通信差异。
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # 所有随机输入均在正式 benchmark 之前生成；不同 rank 使用不同 shard。
        torch.manual_seed(args.seed + 1000 + rank)
        inputs = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(local_batch_size, args.context_length),
            device=device,
        )
        targets = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(local_batch_size, args.context_length),
            device=device,
        )

        parameters = list(model.parameters())
        num_parameters = sum(parameter.numel() for parameter in parameters)
        trainable_parameters = [
            parameter for parameter in parameters if parameter.requires_grad
        ]
        gradient_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in trainable_parameters
        )
        all_reduce_calls = (
            len(trainable_parameters) if args.strategy == "naive" else 1
        )

        def run_step() -> tuple[float, float]:
            optimizer.zero_grad(set_to_none=True)

            # barrier 不计入 iteration latency，只负责对齐各 rank。
            dist.barrier()
            synchronize()
            iteration_start = time.perf_counter()

            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()

            # 隔离 backward 与 gradient synchronization 的 GPU 时间。
            synchronize()
            communication_start = time.perf_counter()
            model.finish_gradient_synchronization()
            synchronize()
            communication_end = time.perf_counter()

            optimizer.step()
            synchronize()
            iteration_end = time.perf_counter()

            return (
                (iteration_end - iteration_start) * 1000,
                (communication_end - communication_start) * 1000,
            )

        for _ in range(args.warmup_steps):
            run_step()

        local_iteration_timings: list[float] = []
        local_communication_timings: list[float] = []

        for _ in range(args.measurement_steps):
            iteration_ms, communication_ms = run_step()
            local_iteration_timings.append(iteration_ms)
            local_communication_timings.append(communication_ms)

        iteration_timings = aggregate_rank_timings(
            local_iteration_timings,
            world_size,
        )
        communication_timings = aggregate_rank_timings(
            local_communication_timings,
            world_size,
        )

        if rank == 0:
            iteration_mean, iteration_std = mean_and_std(iteration_timings)
            communication_mean, communication_std = mean_and_std(
                communication_timings
            )

            row = {
                "hostname": socket.gethostname(),
                "gpu_name": torch.cuda.get_device_name(local_rank),
                "world_size": world_size,
                "strategy": args.strategy,
                "model_size": args.model_size,
                "dtype": args.dtype,
                "global_batch_size": args.global_batch_size,
                "local_batch_size": local_batch_size,
                "context_length": args.context_length,
                "num_parameters": num_parameters,
                "gradient_size_gib": gradient_bytes / 2**30,
                "all_reduce_calls": all_reduce_calls,
                "warmup_steps": args.warmup_steps,
                "measurement_steps": args.measurement_steps,
                "iteration_mean_ms": iteration_mean,
                "iteration_std_ms": iteration_std,
                "gradient_sync_mean_ms": communication_mean,
                "gradient_sync_std_ms": communication_std,
                "communication_percent": (
                    100 * communication_mean / iteration_mean
                ),
            }
            append_csv_row(args.output, row)
            print(json.dumps(row, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
