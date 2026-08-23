from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import statistics
import timeit

import torch
import torch.cuda.nvtx as nvtx
from torch.utils.checkpoint import checkpoint
from einops import einsum

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10B": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
}


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS,
        default="small",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "full"],
        default="full",
    )

    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="启用 Nsight Systems 使用的 NVTX 标记",
    )

    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="记录 CUDA memory snapshot",
    )
    parser.add_argument(
        "--memory-snapshot-path",
        type=Path,
        default=Path("memory_snapshot.pickle"),
    )

    # 可以覆盖预设模型的参数，方便后面的实验。
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)

    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="使用 BF16 autocast",
    )
    parser.add_argument(
        "--checkpoint-block-size",
        type=int,
        default=0,
        help="每个非嵌套 activation checkpoint 包含的 TransformerBlock 数量；0 表示禁用",
    )

    return parser.parse_args()


def resolve_model_config(args: argparse.Namespace) -> dict[str, int]:
    config = MODEL_CONFIGS[args.model_size].copy()

    overrides = {
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
    }

    for name, value in overrides.items():
        if value is not None:
            config[name] = value

    if config["d_model"] % config["num_heads"] != 0:
        raise ValueError("d_model 必须能被 num_heads 整除")

    return config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpointed_transformer_forward(
    model: BasicsTransformerLM,
    token_ids: torch.Tensor,
    block_size: int,
    use_nvtx: bool,
) -> torch.Tensor:
    """Run consecutive, non-nested groups of Transformer blocks under checkpoint."""
    hidden_states = model.token_embeddings(token_ids)

    for start in range(0, len(model.layers), block_size):
        end = min(start + block_size, len(model.layers))

        def run_block(
            block_input: torch.Tensor,
            start_idx: int = start,
            end_idx: int = end,
        ) -> torch.Tensor:
            checkpoint_annotation = (
                nvtx.range(f"checkpoint_blocks_{start_idx}_{end_idx}")
                if use_nvtx
                else nullcontext()
            )
            with checkpoint_annotation:
                block_output = block_input
                for layer_idx in range(start_idx, end_idx):
                    block_output = model.layers[layer_idx](block_output)
                return block_output

        hidden_states = checkpoint(
            run_block,
            hidden_states,
            use_reentrant=False,
        )

    hidden_states = model.ln_final(hidden_states)
    return model.lm_head(hidden_states)


def main() -> None:
    args = parse_args()
    config = resolve_model_config(args)

    def annotation(name: str):
        if args.nvtx:
            return nvtx.range(name)
        return nullcontext()

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    if args.memory_profile and device.type != "cuda":
        raise ValueError("memory profiler 只支持 CUDA")

    if args.mixed_precision and dtype != torch.float32:
        raise ValueError("混合精度实验应保留 FP32 模型参数，请使用 --dtype float32")

    if args.checkpoint_block_size < 0:
        raise ValueError("--checkpoint-block-size 必须大于等于 0")
    if args.checkpoint_block_size > config["num_layers"]:
        raise ValueError("--checkpoint-block-size 不能超过模型层数")
    if args.checkpoint_block_size > 0 and args.mode == "forward":
        raise ValueError("activation checkpointing 仅用于 forward_backward 或 full 模式")

    def precision_context():
        if args.mixed_precision:
            return torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
            )
        return nullcontext()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.nvtx:

        def annotated_scaled_dot_product_attention(
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            with nvtx.range("scaled_dot_product_attention"):
                d_k = K.shape[-1]

                with nvtx.range("attention_qk_matmul"):
                    attention_scores = einsum(
                        Q,
                        K,
                        "... query d_k, ... key d_k -> ... query key",
                    ) / math.sqrt(d_k)

                if mask is not None:
                    with nvtx.range("attention_mask"):
                        attention_scores = torch.where(
                            mask,
                            attention_scores,
                            float("-inf"),
                        )

                with nvtx.range("attention_softmax"):
                    attention_weights = softmax(attention_scores, dim=-1)

                with nvtx.range("attention_pv_matmul"):
                    output = einsum(
                        attention_weights,
                        V,
                        "... query key, ... key d_v -> ... query d_v",
                    )

                return output

        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    # 直接在目标设备上创建模型，避免先在 CPU 分配大模型，
    # 然后再复制到 GPU 所导致的额外内存峰值。
    with torch.device(device):
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
        )

    model = model.to(dtype=dtype)
    model.train(args.mode != "forward")

    # 输入和标签只生成一次，并且放在 GPU 上。
    # 随机数生成和 CPU->GPU 传输不应该算进模型执行时间。
    inputs = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    def model_forward(token_ids: torch.Tensor) -> torch.Tensor:
        if args.checkpoint_block_size == 0:
            return model(token_ids)
        return checkpointed_transformer_forward(
            model=model,
            token_ids=token_ids,
            block_size=args.checkpoint_block_size,
            use_nvtx=args.nvtx,
        )

    def prepare_step() -> None:
        # 清除上一轮梯度，但不把它计入 forward/backward 时间。
        if args.mode != "forward":
            optimizer.zero_grad(set_to_none=True)

    def run_step() -> torch.Tensor:
        if args.mode == "forward":
            with annotation("forward"):
                with torch.no_grad(), precision_context():
                    return model_forward(inputs)

        with annotation("forward"):
            with precision_context():
                logits = model_forward(inputs)
                loss = cross_entropy(logits, targets)

        with annotation("backward"):
            loss.backward()

        if args.mode == "full":
            with annotation("optimizer_step"):
                optimizer.step()

        return loss

    # Warm-up 阶段：执行相同的工作，但不记录时间。
    for _ in range(args.warmup_steps):
        prepare_step()
        value = run_step()
        synchronize(device)
        del value

    timings_seconds: list[float] = []

    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.memory_profile:
        args.memory_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    try:
        with annotation("measurement"):
            for _ in range(args.measurement_steps):
                prepare_step()

                # 确保前面提交的 CUDA 工作已经完成。
                synchronize(device)
                start = timeit.default_timer()

                value = run_step()

                # CUDA 是异步的。必须等待 GPU 真正执行完成后再停止计时。
                synchronize(device)
                end = timeit.default_timer()

                timings_seconds.append(end - start)
                del value
    finally:
        if args.memory_profile:
            synchronize(device)
            try:
                torch.cuda.memory._dump_snapshot(str(args.memory_snapshot_path))
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)

    timings_ms = [value * 1000 for value in timings_seconds]
    mean_ms = statistics.fmean(timings_ms)
    std_ms = statistics.stdev(timings_ms) if len(timings_ms) >= 2 else 0.0

    result = {
        "model_size": args.model_size,
        "mode": args.mode,
        "dtype": args.dtype,
        "mixed_precision": args.mixed_precision,
        "checkpoint_block_size": args.checkpoint_block_size,
        "num_checkpoint_segments": (
            math.ceil(config["num_layers"] / args.checkpoint_block_size)
            if args.checkpoint_block_size > 0
            else 0
        ),
        "device": str(device),
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "peak_memory_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        "peak_memory_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        "memory_snapshot_path": (
            str(args.memory_snapshot_path)
            if args.memory_profile
            else None
        ),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "timings_ms": timings_ms,
        **config,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
