#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

output="${OUTPUT:-profiles/optimizer_state_sharding_accounting.csv}"
nproc_per_node="${NPROC_PER_NODE:-2}"
model_size="${MODEL_SIZE:-xl}"
global_batch_size="${GLOBAL_BATCH_SIZE:-4}"
context_length="${CONTEXT_LENGTH:-512}"
warmup_steps="${WARMUP_STEPS:-2}"
measurement_steps="${MEASUREMENT_STEPS:-5}"

common_args=(
    --model-size "$model_size"
    --global-batch-size "$global_batch_size"
    --context-length "$context_length"
    --warmup-steps "$warmup_steps"
    --measurement-steps "$measurement_steps"
    --output "$output"
)

printf '===== Full AdamW state on every rank =====\n' >&2
uv run torchrun \
    --standalone \
    --nproc-per-node "$nproc_per_node" \
    -m cs336_systems.optimizer_state_sharding_benchmark \
    --optimizer-strategy baseline \
    "${common_args[@]}" \
    "$@"

printf '\n===== Sharded AdamW state =====\n' >&2
uv run torchrun \
    --standalone \
    --nproc-per-node "$nproc_per_node" \
    -m cs336_systems.optimizer_state_sharding_benchmark \
    --optimizer-strategy sharded \
    --append-output \
    "${common_args[@]}" \
    "$@"

printf '\nCSV written to %s\n' "$output" >&2
