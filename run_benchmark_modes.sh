#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

for arg in "$@"; do
    if [[ "$arg" == "--mode" || "$arg" == --mode=* ]]; then
        printf '错误：run_benchmark_modes.sh 会自动运行三个 mode，请不要手动传入 --mode。\n' >&2
        exit 2
    fi
done

modes=(
    forward
    forward_backward
    full
)

for mode in "${modes[@]}"; do
    printf '\n===== mode: %s =====\n' "$mode" >&2
    uv run python -m cs336_systems.benchmark \
        --mode "$mode" \
        "$@"
done
