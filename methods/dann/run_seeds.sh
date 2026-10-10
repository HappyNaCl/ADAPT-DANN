#!/usr/bin/env bash
#
# Run dann.py with every default parameter kept,
# varying only the seed: 42, 84, 123, 256, 512.
# Runs in quiet mode (--use_verbose false): each log holds just the summary.
#
# Each seed writes its own checkpoint + metrics to outputs/seed_<SEED>/,
# so runs don't overwrite each other.
#
# Usage:
#   ./run_seeds.sh
#
# Extra args are forwarded to dann.py, e.g.:
#   ./run_seeds.sh --use_adapters --adapter_dim 32
#
set -euo pipefail

cd "$(dirname "$0")"

SEEDS=(42 84 123 256 512)
SCRIPT="dann.py"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

for SEED in "${SEEDS[@]}"; do
    LOG="${LOG_DIR}/seed_${SEED}.log"
    : > "$LOG"

    {
        echo "=============================================================="
        echo " Running with seed: ${SEED}"
        echo "=============================================================="
    } | tee -a "$LOG"

    # NOTE: script path first, THEN its arguments.
    python3 "$SCRIPT" \
        --seed "$SEED" \
        --output_dir "outputs/seed_${SEED}" \
        --use_verbose false \
        "$@" \
        2>&1 | tee -a "$LOG"
done

echo "All seeds finished: ${SEEDS[*]}"
