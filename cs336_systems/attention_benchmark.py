from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import statistics
import timeit

import torch

from cs336_basics.model import scaled_dot_product_attention


BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQUENCE_LENGTHS = [256, 1024, 4096, 8192, 16384]


def synchronize() -> None:
    torch.cuda.synchronize()


def mean_and_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean, std


def benchmark_configuration(
    d_model: int,
    sequence_length: int,
    warmup_steps: int,
    repetitions: int,
) -> dict[str, float | int | str]:
    device = torch.device("cuda")
    shape = (BATCH_SIZE, sequence_length, d_model)

    q = torch.randn(shape, device=device, requires_grad=True)
    k = torch.randn(shape, device=device, requires_grad=True)
    v = torch.randn(shape, device=device, requires_grad=True)
    grad_output = torch.randn(shape, device=device)

    # Forward warm-up.
    for _ in range(warmup_steps):
        output = scaled_dot_product_attention(q, k, v)
        synchronize()
        del output

    forward_times_ms: list[float] = []

    for _ in range(repetitions):
        synchronize()
        start = timeit.default_timer()

        output = scaled_dot_product_attention(q, k, v)

        synchronize()
        end = timeit.default_timer()

        forward_times_ms.append((end - start) * 1000)
        del output

    # Backward warm-up。每次都创建新的 autograd graph。
    for _ in range(warmup_steps):
        q.grad = None
        k.grad = None
        v.grad = None

        output = scaled_dot_product_attention(q, k, v)
        synchronize()

        output.backward(grad_output)
        synchronize()
        del output

    backward_times_ms: list[float] = []
    memory_before_backward_mib: list[float] = []
    saved_for_backward_mib: list[float] = []

    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(repetitions):
        q.grad = None
        k.grad = None
        v.grad = None
        synchronize()

        # 输入 tensor 和 grad_output 的基础显存。
        memory_before_forward = torch.cuda.memory_allocated(device)

        # 创建本轮 backward 所需的新计算图，但不计入 backward 时间。
        output = scaled_dot_product_attention(q, k, v)
        synchronize()

        memory_before_backward = torch.cuda.memory_allocated(device)
        memory_before_backward_mib.append(
            memory_before_backward / 2**20
        )
        saved_for_backward_mib.append(
            (memory_before_backward - memory_before_forward) / 2**20
        )

        start = timeit.default_timer()

        output.backward(grad_output)

        synchronize()
        end = timeit.default_timer()

        backward_times_ms.append((end - start) * 1000)
        del output

    forward_mean, forward_std = mean_and_std(forward_times_ms)
    backward_mean, backward_std = mean_and_std(backward_times_ms)

    return {
        "status": "ok",
        "batch_size": BATCH_SIZE,
        "sequence_length": sequence_length,
        "d_model": d_model,
        "forward_mean_ms": forward_mean,
        "forward_std_ms": forward_std,
        "backward_mean_ms": backward_mean,
        "backward_std_ms": backward_std,
        "memory_before_backward_mib": statistics.fmean(
            memory_before_backward_mib
        ),
        "saved_for_backward_mib": statistics.fmean(
            saved_for_backward_mib
        ),
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
        ),
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "status",
        "batch_size",
        "sequence_length",
        "d_model",
        "forward_mean_ms",
        "forward_std_ms",
        "backward_mean_ms",
        "backward_std_ms",
        "memory_before_backward_mib",
        "saved_for_backward_mib",
        "peak_memory_mib",
        "error",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--d-models",
        type=int,
        nargs="+",
        default=D_MODELS,
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=SEQUENCE_LENGTHS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/attention_benchmark.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    rows: list[dict] = []

    for sequence_length in args.sequence_lengths:
        for d_model in args.d_models:
            try:
                result = benchmark_configuration(
                    d_model=d_model,
                    sequence_length=sequence_length,
                    warmup_steps=args.warmup_steps,
                    repetitions=args.repetitions,
                )
            except torch.OutOfMemoryError as error:
                result = {
                    "status": "OOM",
                    "batch_size": BATCH_SIZE,
                    "sequence_length": sequence_length,
                    "d_model": d_model,
                    "error": str(error).splitlines()[0],
                }

            rows.append(result)
            print(json.dumps(result, indent=2))

            # 每个配置后立即保存，避免长时间运行中途退出丢失结果。
            write_csv(rows, args.output)

            gc.collect()
            torch.cuda.empty_cache()

    print(f"结果已保存至 {args.output}")


if __name__ == "__main__":
    main()