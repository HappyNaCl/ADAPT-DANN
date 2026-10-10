#!/usr/bin/env bash
#
# Run adapter_v2_with_tsne.py with every default parameter kept,
# varying only the seed: 42, 84, 123, 256, 512.
# Runs in quiet mode (--use_verbose false): each log holds just the summary.
#
# Usage:
#   ./run_seeds.sh
#
set -euo pipefail

cd "$(dirname "$0")"

SEEDS=(42 84 123 256 512)
SCRIPT="adapter_v2_with_tsne.py"
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
    python3 "$SCRIPT" --seed "$SEED" --use_verbose false 2>&1 | tee -a "$LOG"
done

echo "All seeds finished: ${SEEDS[*]}"
