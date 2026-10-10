#!/usr/bin/env bash
#
# Run adapt-dann-v2.py with every default parameter kept,
# varying only the seed: 42, 84, 123, 256, 512.
# Quiet mode: each log holds just the summary (+ domain_probe line).
#
# Usage:
#   ./run_seeds_v2.sh                         # all seeds
#   SEEDS="42 84" ./run_seeds_v2.sh           # subset
#   ./run_seeds_v2.sh --target_entropy_weight 0.1   # extra args go to the script
#
set -uo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-42 84 123 256 512}"
SCRIPT="adapt-dann-v2.py"
LOG_DIR="outputs/adapt_dann_v2"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary_all.txt"
: > "$SUMMARY"

failed=()
for SEED in $SEEDS; do
    LOG="$LOG_DIR/seed_${SEED}.log"
    echo "=== adapt-dann-v2 seed=${SEED} ==="
    t0=$(date +%s)
    "$PYTHON" "$SCRIPT" --seed "$SEED" "$@" > "$LOG" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    {
        echo "### seed=${SEED}  (rc=${rc}, ${dt}s)"
        grep -E '^(Model:|batch=|accuracy:|precision:|recall:|f1:|domain_probe:)' "$LOG"
        echo
    } >> "$SUMMARY"
    if [ "$rc" -ne 0 ]; then
        echo "!! seed=${SEED} FAILED (rc=${rc}) — see $LOG"
        tail -n 15 "$LOG" | sed 's/^/   | /'
        failed+=("$SEED")
    else
        grep -E '^(f1:|domain_probe:)' "$LOG"
    fi
done

echo "Done. Summaries: $SUMMARY"
[ ${#failed[@]} -eq 0 ] || { echo "Failed seeds: ${failed[*]}"; exit 1; }
