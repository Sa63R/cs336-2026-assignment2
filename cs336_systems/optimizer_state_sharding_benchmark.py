from __future__ import annotations

import argparse
from collections.abc import Iterable
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp import OverlapDDP
from cs336_systems.sharded_optimizer import ShardedOptimizer


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

CSV_FIELDS = [
    "hostname",
    "gpu_name",
    "world_size",
    "optimizer_strategy",
    "model_size",
    "dtype",
    "global_batch_size",
    "local_batch_size",
    "context_length",
    "vocab_size",
    "num_parameters",
    "parameter_memory_mib",
    "buffer_memory_mib",
    "gradient_memory_rank_min_mib",
    "gradient_memory_rank_max_mib",
    "optimizer_state_memory_rank_min_mib",
    "optimizer_state_memory_rank_max_mib",
    "optimizer_owned_parameters_rank_min",
    "optimizer_owned_parameters_rank_max",
    "model_init_allocated_rank_min_mib",
    "model_init_allocated_rank_max_mib",
    "model_init_peak_rank_min_mib",
    "model_init_peak_rank_max_mib",
    "before_optimizer_step_allocated_rank_min_mib",
    "before_optimizer_step_allocated_rank_max_mib",
    "before_optimizer_step_peak_rank_min_mib",
    "before_optimizer_step_peak_rank_max_mib",
    "after_optimizer_step_allocated_rank_min_mib",
    "after_optimizer_step_allocated_rank_max_mib",
    "after_optimizer_step_peak_rank_min_mib",
    "after_optimizer_step_peak_rank_max_mib",
    "warmup_steps",
    "measurement_steps",
    "iteration_mean_ms",
    "iteration_std_ms",
    "timings_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full and sharded AdamW optimizer state memory and training latency.",
    )
    parser.add_argument(
        "--optimizer-strategy",
        choices=["baseline", "sharded"],
        required=True,
    )
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/optimizer_state_sharding_accounting.csv"),
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append one row instead of replacing the output CSV.",
    )
    return parser.parse_args()


def memory_snapshot(device: torch.device, prefix: str) -> dict[str, float]:
    """Read current allocated memory and the peak since the last reset."""

    torch.cuda.synchronize(device)
    return {
        f"{prefix}_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
        f"{prefix}_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Count only CUDA tensors; Adam's CPU step counters are not GPU memory."""

    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                total += value.numel() * value.element_size()
    return total


def locally_owned_parameter_count(optimizer: torch.optim.Optimizer) -> int:
    if isinstance(optimizer, ShardedOptimizer):
        local_optimizer = optimizer.local_optimizer
        if local_optimizer is None:
            return 0
        groups = local_optimizer.param_groups
    else:
        groups = optimizer.param_groups
    return sum(parameter.numel() for group in groups for parameter in group["params"])


def gather_rank_metrics(
    local_metrics: dict[str, float | int],
    world_size: int,
) -> list[dict[str, float | int]]:
    gathered: list[dict[str, float | int] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_metrics)
    return [metrics for metrics in gathered if metrics is not None]


def aggregate_rank_timings(local_timings: list[float], world_size: int) -> list[float]:
    gathered: list[list[float] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_timings)

    # A distributed iteration completes only when its slowest rank completes.
    return [max(rank_timings[step] for rank_timings in gathered if rank_timings is not None) for step in range(len(local_timings))]


def add_rank_range(
    result: dict[str, Any],
    rank_metrics: list[dict[str, float | int]],
    metric_name: str,
    output_name: str,
) -> None:
    values = [metrics[metric_name] for metrics in rank_metrics]
    if output_name.endswith("_mib"):
        stem = output_name.removesuffix("_mib")
        result[f"{stem}_rank_min_mib"] = min(values)
        result[f"{stem}_rank_max_mib"] = max(values)
    else:
        result[f"{output_name}_rank_min"] = min(values)
        result[f"{output_name}_rank_max"] = max(values)


def write_csv(path: Path, row: dict[str, Any], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open(mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.global_batch_size % world_size != 0:
        raise ValueError("--global-batch-size must be divisible by WORLD_SIZE")
    if world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, but only {torch.cuda.device_count()} CUDA devices are visible",
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
        local_batch_size = args.global_batch_size // world_size

        # Reset before constructing the model so the first peak covers model
        # initialization and the DDP rank-0 parameter broadcast only.
        torch.cuda.reset_peak_memory_stats(device)
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
        model = OverlapDDP(base_model)
        model.train()

        model_init_memory = memory_snapshot(device, "model_init")
        torch.cuda.reset_peak_memory_stats(device)

        if args.optimizer_strategy == "baseline":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
            )
        else:
            optimizer = ShardedOptimizer(
                model.parameters(),
                torch.optim.AdamW,
                lr=args.learning_rate,
            )

        # Input creation is intentionally outside every timed iteration.
        torch.manual_seed(args.seed + 1000 + rank)
        inputs = torch.randint(
            0,
            args.vocab_size,
            (local_batch_size, args.context_length),
            device=device,
        )
        targets = torch.randint(
            0,
            args.vocab_size,
            (local_batch_size, args.context_length),
            device=device,
        )

        parameters = list(model.parameters())
        buffers = list(model.buffers())
        num_parameters = sum(parameter.numel() for parameter in parameters)

        # The first training step is the accounting step. AdamW creates its
        # exp_avg and exp_avg_sq tensors lazily in optimizer.step(), which is
        # exactly why memory must be observed on both sides of that call.
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        model.finish_gradient_synchronization()
        del logits, loss

        before_step_memory = memory_snapshot(device, "before_optimizer_step")
        local_gradient_bytes = tensor_bytes(
            [parameter.grad for parameter in parameters if parameter.grad is not None],
        )

        # Start a new peak interval at the current pre-step allocation. The
        # following peak therefore includes Adam state creation and temporary
        # foreach buffers, but not an earlier forward activation peak.
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.step()
        after_step_memory = memory_snapshot(device, "after_optimizer_step")

        local_metrics: dict[str, float | int] = {
            **model_init_memory,
            **before_step_memory,
            **after_step_memory,
            "gradient_memory_mib": local_gradient_bytes / 2**20,
            "optimizer_state_memory_mib": optimizer_state_bytes(optimizer) / 2**20,
            "optimizer_owned_parameters": locally_owned_parameter_count(optimizer),
        }
        rank_metrics = gather_rank_metrics(local_metrics, world_size)

        def run_training_step() -> float:
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize(device)
            start = time.perf_counter()

            step_logits = model(inputs)
            step_loss = cross_entropy(step_logits, targets)
            step_loss.backward()
            model.finish_gradient_synchronization()
            optimizer.step()

            torch.cuda.synchronize(device)
            end = time.perf_counter()
            del step_logits, step_loss
            return (end - start) * 1000

        for _ in range(args.warmup_steps):
            run_training_step()

        local_timings = [run_training_step() for _ in range(args.measurement_steps)]
        timings = aggregate_rank_timings(local_timings, world_size)

        if rank == 0:
            result: dict[str, Any] = {
                "hostname": socket.gethostname(),
                "gpu_name": torch.cuda.get_device_name(local_rank),
                "world_size": world_size,
                "optimizer_strategy": args.optimizer_strategy,
                "model_size": args.model_size,
                "dtype": "float32",
                "global_batch_size": args.global_batch_size,
                "local_batch_size": local_batch_size,
                "context_length": args.context_length,
                "vocab_size": args.vocab_size,
                "num_parameters": num_parameters,
                "parameter_memory_mib": tensor_bytes(parameters) / 2**20,
                "buffer_memory_mib": tensor_bytes(buffers) / 2**20,
                "warmup_steps": args.warmup_steps,
                "measurement_steps": args.measurement_steps,
                "iteration_mean_ms": statistics.fmean(timings),
                "iteration_std_ms": statistics.stdev(timings) if len(timings) >= 2 else 0.0,
                "timings_ms": json.dumps(timings),
            }

            for metric_name in (
                "gradient_memory_mib",
                "optimizer_state_memory_mib",
                "optimizer_owned_parameters",
                "model_init_allocated_mib",
                "model_init_peak_mib",
                "before_optimizer_step_allocated_mib",
                "before_optimizer_step_peak_mib",
                "after_optimizer_step_allocated_mib",
                "after_optimizer_step_peak_mib",
            ):
                add_rank_range(result, rank_metrics, metric_name, metric_name)

            write_csv(args.output, result, append=args.append_output)
            print(json.dumps(result, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
