#!/usr/bin/env bash
#
# run_seeds.sh — train ADAPT-DANN with every default parameter kept,
# varying only the seed: 42, 84, 123, 256, 512.
#
# Quiet mode (default): each log holds just the compact header + summary.
#
# Usage:
#   ./run_seeds.sh                              # seeds 42 84 123 256 512
#   ./run_seeds.sh --epochs_task 5 --batch_size 64   # extra args go to adapt_dann.py
#   SEEDS="42 84" ./run_seeds.sh                # subset of seeds
#   VERBOSE=1 ./run_seeds.sh                    # full training logs in each log file
#   PYTHON=python3.8 ./run_seeds.sh             # choose the interpreter
#
# Outputs (per seed):
#   logs/seed_<N>.log               run log (header + summary unless VERBOSE=1)
#   outputs/seed_<N>/               adapters, checkpoints, plots
#   logs/summary.txt                all per-seed summaries in one file
#   logs/failures.txt               failed runs (if any)
#
set -uo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-42 84 123 256 512}"
SCRIPT="adapt_dann.py"
LOG_DIR="logs"
OUT_ROOT="outputs"

# Never block on plt.show(); silence the HF forked-process warning.
export MPLBACKEND="${MPLBACKEND:-Agg}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

declare -a EXTRA_ARGS=()
if [ "${VERBOSE:-0}" = "1" ] || [ "${VERBOSE:-0}" = "true" ]; then
  EXTRA_ARGS+=(--use-verbose)
fi
EXTRA_ARGS+=("$@")

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: $SCRIPT not found in $(pwd)" >&2
  exit 1
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: interpreter '$PYTHON' not found (set PYTHON=... to override)" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
SUMMARY_FILE="$LOG_DIR/summary.txt"
FAIL_FILE="$LOG_DIR/failures.txt"
: > "$SUMMARY_FILE"
: > "$FAIL_FILE"

total=0
passed=0
failed=0
started=$(date +%s)

printf 'Seed loop: %s\n' "$SEEDS"
printf 'Logs:      %s\n\n' "$LOG_DIR"

for SEED in $SEEDS; do
  total=$((total + 1))
  OUT_DIR="$OUT_ROOT/seed_${SEED}"
  LOG="$LOG_DIR/seed_${SEED}.log"

  printf '[%2d] seed=%-4s ... ' "$total" "$SEED"

  t0=$(date +%s)
  "$PYTHON" "$SCRIPT" \
    --seed "$SEED" \
    --output_dir "$OUT_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "$LOG" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))

  # Pull the compact summary out of the log (quiet mode => stdout is just that)
  {
    echo "### ADAPT-DANN seed=$SEED  (rc=$rc, ${dt}s)"
    grep -E '^(accuracy|precision|recall|f1): source=' -B2 "$LOG" | grep -v '^--$'
    echo
  } >> "$SUMMARY_FILE"

  if [ $rc -eq 0 ]; then
    passed=$((passed + 1))
    f1=$(grep -E '^f1: source=' "$LOG" | tail -n 1 | grep -o 'target=[0-9.]*' | cut -d= -f2)
    printf 'ok  (%4ss)  target_f1=%-8s -> %s\n' "$dt" "${f1:-n/a}" "$LOG"
  else
    failed=$((failed + 1))
    printf 'FAILED rc=%s (%4ss)  -> %s\n' "$rc" "$dt" "$LOG"
    echo "ADAPT-DANN seed=$SEED rc=$rc log=$LOG" >> "$FAIL_FILE"
    # keep the tail of the log visible on the terminal for quick diagnosis
    tail -n 15 "$LOG" | sed 's/^/      | /'
  fi
done

elapsed=$(( $(date +%s) - started ))

printf '\n%s\n' '------------------------------------------------------------'
printf 'Done in %dm%02ds — total=%d  ok=%d  failed=%d\n' \
       $((elapsed / 60)) $((elapsed % 60)) "$total" "$passed" "$failed"
printf 'Summaries: %s\n' "$SUMMARY_FILE"
if [ "$failed" -gt 0 ]; then
  printf 'Failures:  %s\n' "$FAIL_FILE"
  cat "$FAIL_FILE"
fi
printf '%s\n' '------------------------------------------------------------'

[ "$failed" -eq 0 ] || exit 1
