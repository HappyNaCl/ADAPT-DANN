#!/usr/bin/env bash
#
# Run fine_tune_dapt_source.py over seeds 42, 84, 123, 256, 512.
# Only --seed changes between runs. Quiet by default (summary only).
#
# Usage:
#   ./run_dapt_source_seeds.sh                 # quiet: summary only
#   ./run_dapt_source_seeds.sh --use-verbose   # full training logs
#   SEEDS="42 84" ./run_dapt_source_seeds.sh   # subset of seeds
#
# Logs: outputs/seed_logs/seed_<N>.log
# All summaries collected in outputs/seed_logs/summary_all.txt

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"   # script uses relative paths

PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-42 84 123 256 512}"
SCRIPT="fine_tune_dapt_source.py"
LOG_DIR="outputs/seed_logs"
EXTRA_ARGS=("$@")                     # e.g. --use-verbose

export MPLBACKEND="${MPLBACKEND:-Agg}"                       # don't block on plt.show()
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary_all.txt"
: > "$SUMMARY"

failed=()
for SEED in $SEEDS; do
    LOG="$LOG_DIR/seed_${SEED}.log"
    echo "=============================================================="
    echo " fine_tune_dapt_source.py  seed=${SEED}"
    echo "=============================================================="

    t0=$(date +%s)
    "$PYTHON" "$SCRIPT" --seed "$SEED" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    dt=$(( $(date +%s) - t0 ))

    {
        echo "### seed=${SEED}  (rc=${rc}, ${dt}s)"
        grep -E '^(Model:|batch=|accuracy:|precision:|recall:|f1:)' "$LOG"
        echo
    } >> "$SUMMARY"

    if [ "$rc" -ne 0 ]; then
        echo "!! seed=${SEED} FAILED (rc=${rc}) — continuing"
        failed+=("$SEED")
    fi
done

echo
echo "Done. Summaries: $SUMMARY"
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed seeds: ${failed[*]}"
    exit 1
fi
