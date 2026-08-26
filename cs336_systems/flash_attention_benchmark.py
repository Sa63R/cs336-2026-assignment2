from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import triton

# 使用课程中的朴素 attention。
# 不要使用 torch.nn.functional.scaled_dot_product_attention，
# 因为它可能自动调用 PyTorch 自带的 FlashAttention。
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention import FlashAttention2Triton


BATCH_SIZE = 1
CAUSAL = True

SEQUENCE_LENGTHS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]

D_MODELS = [16, 32, 64, 128]

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

IMPLEMENTATIONS = ["pytorch", "flash"]

MODES = [
    "forward",
    "backward",
    "forward_backward",
]

CSV_FIELDS = [
    "gpu",
    "batch_size",
    "causal",
    "implementation",
    "dtype",
    "sequence_length",
    "d_model",
    "forward_status",
    "forward_ms",
    "forward_error",
    "backward_status",
    "backward_ms",
    "backward_error",
    "forward_backward_status",
    "forward_backward_ms",
    "forward_backward_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=SEQUENCE_LENGTHS,
    )
    parser.add_argument(
        "--d-models",
        type=int,
        nargs="+",
        default=D_MODELS,
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DTYPES,
        default=list(DTYPES),
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=IMPLEMENTATIONS,
        default=IMPLEMENTATIONS,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES,
    )

    # 注意：do_bench 的 warmup 和 rep 单位都是毫秒。
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "profiles/flash_attention_benchmark.csv"
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
            extrasaction="ignore",
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def load_completed_configurations(
    output_path: Path,
) -> set[tuple[str, str, int, int]]:
    """
    读取已有 CSV，使 benchmark 中断后可以继续运行，
    避免重复测试已经完成的配置。
    """
    if not output_path.exists():
        return set()

    completed = set()

    with output_path.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            completed.add(
                (
                    row["implementation"],
                    row["dtype"],
                    int(row["sequence_length"]),
                    int(row["d_model"]),
                )
            )

    return completed


def clear_gradients(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> None:
    Q.grad = None
    K.grad = None
    V.grad = None


def benchmark_mode(
    implementation: str,
    mode: str,
    dtype: torch.dtype,
    sequence_length: int,
    d_model: int,
    warmup_ms: int,
    rep_ms: int,
    seed: int,
) -> float:
    """
    返回 do_bench 测得的 median latency，单位为毫秒。
    所有输入都在正式 benchmark 之前生成。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")

    shape = (
        BATCH_SIZE,
        sequence_length,
        d_model,
    )

    Q = torch.randn(
        shape,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    K = torch.randn(
        shape,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    V = torch.randn(
        shape,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dO = torch.randn(
        shape,
        device=device,
        dtype=dtype,
    )

    if implementation == "pytorch":
        # mask 也在正式计时前创建。
        positions = torch.arange(
            sequence_length,
            device=device,
        )
        causal_mask = (
            positions[:, None]
            >= positions[None, :]
        )

        def attention() -> torch.Tensor:
            return scaled_dot_product_attention(
                Q,
                K,
                V,
                causal_mask,
            )

    elif implementation == "flash":

        def attention() -> torch.Tensor:
            return FlashAttention2Triton.apply(
                Q,
                K,
                V,
                True,
            )

    else:
        raise ValueError(
            f"Unknown implementation: {implementation}"
        )

    if mode == "forward":

        def run() -> None:
            # 输入 requires_grad=True，所以这里测的是训练时 forward。
            attention()

        grad_to_none = None

    elif mode == "backward":
        # 创建 backward 使用的计算图，但不把 forward 算入时间。
        output = attention()

        def run() -> None:
            output.backward(
                dO,
                retain_graph=True,
            )

        grad_to_none = [Q, K, V]

    elif mode == "forward_backward":

        def run() -> None:
            # 每次都重新 forward，创建新的 autograd graph。
            output = attention()
            output.backward(dO)

        grad_to_none = [Q, K, V]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 提前触发 Triton JIT 和 torch.compile。
    # 编译时间不能计入正式 latency。
    run()
    torch.cuda.synchronize()
    clear_gradients(Q, K, V)

    latency_ms = triton.testing.do_bench(
        run,
        warmup=warmup_ms,
        rep=rep_ms,
        grad_to_none=grad_to_none,
        return_mode="median",
    )

    return float(latency_ms)


def benchmark_mode_safely(
    **kwargs,
) -> tuple[str, float | None, str]:
    try:
        latency_ms = benchmark_mode(**kwargs)
        return "ok", latency_ms, ""

    except Exception as error:
        message = (
            f"{type(error).__name__}: {error}"
        ).splitlines()[0]

        if (
            isinstance(error, torch.OutOfMemoryError)
            or "out of memory" in message.lower()
        ):
            status = "OOM"
        else:
            status = "ERROR"

        return status, None, message

    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # 与 Triton kernel 的 input_precision='ieee' 对齐。
    torch.set_float32_matmul_precision("highest")

    gpu_name = torch.cuda.get_device_name()
    completed = load_completed_configurations(
        args.output
    )

    for sequence_length in args.sequence_lengths:
        for d_model in args.d_models:
            for dtype_name in args.dtypes:
                dtype = DTYPES[dtype_name]

                for implementation in args.implementations:
                    configuration = (
                        implementation,
                        dtype_name,
                        sequence_length,
                        d_model,
                    )

                    if configuration in completed:
                        print(
                            "Skipping completed configuration:",
                            configuration,
                        )
                        continue

                    row: dict[str, object] = {
                        "gpu": gpu_name,
                        "batch_size": BATCH_SIZE,
                        "causal": CAUSAL,
                        "implementation": implementation,
                        "dtype": dtype_name,
                        "sequence_length": sequence_length,
                        "d_model": d_model,
                    }

                    for mode in args.modes:
                        status, latency_ms, error = (
                            benchmark_mode_safely(
                                implementation=implementation,
                                mode=mode,
                                dtype=dtype,
                                sequence_length=sequence_length,
                                d_model=d_model,
                                warmup_ms=args.warmup_ms,
                                rep_ms=args.rep_ms,
                                seed=args.seed,
                            )
                        )

                        row[f"{mode}_status"] = status
                        row[f"{mode}_ms"] = latency_ms
                        row[f"{mode}_error"] = error

                        print(
                            json.dumps(
                                {
                                    "implementation": implementation,
                                    "dtype": dtype_name,
                                    "sequence_length": (
                                        sequence_length
                                    ),
                                    "d_model": d_model,
                                    "mode": mode,
                                    "status": status,
                                    "latency_ms": latency_ms,
                                    "error": error,
                                },
                                indent=2,
                            )
                        )

                    # 每个配置完成后立即保存，避免长时间运行丢数据。
                    append_csv_row(args.output, row)
                    completed.add(configuration)

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()