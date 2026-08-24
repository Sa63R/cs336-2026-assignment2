#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

attention_csv="${ATTENTION_CSV:-profiles/attention_compile_benchmark.csv}"
transformer_csv="${TRANSFORMER_CSV:-profiles/transformer_compile_benchmark.csv}"
run_attention="${RUN_ATTENTION:-1}"
run_transformer="${RUN_TRANSFORMER:-1}"
attention_warmup_steps="${ATTENTION_WARMUP_STEPS:-10}"
attention_repetitions="${ATTENTION_REPETITIONS:-100}"
read -r -a attention_d_models <<< "${ATTENTION_D_MODELS:-16 32 64 128}"
read -r -a attention_sequence_lengths <<< "${ATTENTION_SEQUENCE_LENGTHS:-256 1024 4096 8192 16384}"
model_size="${MODEL_SIZE:-small}"
batch_size="${BATCH_SIZE:-4}"
context_length="${CONTEXT_LENGTH:-512}"
warmup_steps="${WARMUP_STEPS:-5}"
measurement_steps="${MEASUREMENT_STEPS:-10}"

if [[ "$run_attention" == 1 ]]; then
    printf '===== Attention: uncompiled vs compiled =====\n' >&2
    uv run python -m cs336_systems.attention_benchmark \
        --implementations uncompiled compiled \
        --d-models "${attention_d_models[@]}" \
        --sequence-lengths "${attention_sequence_lengths[@]}" \
        --warmup-steps "$attention_warmup_steps" \
        --repetitions "$attention_repetitions" \
        --output "$attention_csv"
fi

first_transformer_result=true

if [[ "$run_transformer" == 1 ]]; then
    for mode in forward full; do
        for implementation in vanilla compiled; do
            compile_args=()
            if [[ "$implementation" == "compiled" ]]; then
                compile_args+=(--compile-model)
            fi

            csv_args=(--csv-output "$transformer_csv")
            if [[ "$first_transformer_result" == false ]]; then
                csv_args+=(--append-csv)
            fi

            printf '\n===== Transformer: mode=%s implementation=%s =====\n' \
                "$mode" "$implementation" >&2

            uv run python -m cs336_systems.benchmark \
                --model-size "$model_size" \
                --batch-size "$batch_size" \
                --context-length "$context_length" \
                --mode "$mode" \
                --dtype float32 \
                --warmup-steps "$warmup_steps" \
                --measurement-steps "$measurement_steps" \
                "${compile_args[@]}" \
                "${csv_args[@]}"

            first_transformer_result=false
        done
    done
fi

if [[ "$run_attention" == 1 ]]; then
    printf '\nAttention CSV: %s\n' "$attention_csv" >&2
fi
if [[ "$run_transformer" == 1 ]]; then
    printf 'Transformer CSV: %s\n' "$transformer_csv" >&2
fi
