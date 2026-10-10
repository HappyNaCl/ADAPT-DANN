#!/usr/bin/env bash
#
# run_seed_loop.sh — run every training method over a fixed set of seeds.
# The ONLY thing that changes between runs is --seed.
#
# Usage:
#   ./run_seed_loop.sh                          # all 6 methods, seeds 42 84 123 256 512
#   SEEDS="42 84" ./run_seed_loop.sh            # subset of seeds
#   ./run_seed_loop.sh IndoBERT ADAPT-DANN      # subset of methods (space-separated names)
#   PYTHON=python3.8 ./run_seed_loop.sh         # choose the interpreter
#
# Output (per run):  results/seed_loop/<method>_seed<N>.log
#   Because every script defaults to quiet mode, the .log holds just the
#   compact 2-line header + the summary lines, plus any library stderr.
# Summaries also get appended to results/seed_loop/summary_all.txt
# with a "### <method> seed=<N>" marker.
# A failed run is recorded in results/seed_loop/failures.txt and the loop
# continues; a pass/fail tally is printed at the end.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-42 84 123 256 512}"
OUT_DIR="$ROOT/results/seed_loop"

# Never block on plt.show() (t-SNE / learning-curve plots).
export MPLBACKEND="${MPLBACKEND:-Agg}"
# Silence the "process just got forked ... Disabling parallelism" HF warning.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$OUT_DIR"
SUMMARY_FILE="$OUT_DIR/summary_all.txt"
FAIL_FILE="$OUT_DIR/failures.txt"
: > "$SUMMARY_FILE"
: > "$FAIL_FILE"

# name|relative dir|script  (each script must run from its own directory,
# because they use ../../datasets/... and ./models/... relative paths)
METHODS_ALL=(
  "IndoBERT|methods/IndoBERT|fine_tune_target_direct.py"
  "DAPT-source-only|methods/pretraining|fine_tune_source_only.py"
  "DAPT-finetune-target|methods/pretraining|fine_tune_source.py"
  "DAPT-no-adapter|methods/DAPT + Adapter|fine_tune_dapt_source.py"
  "DAPT+Adapter|methods/DAPT + Adapter|fine_tune_dapt_adapter_source.py"
  "ADAPT-DANN|methods/ADAPT-DANN|adapt_dann.py"
)

# Optional name filter from the command line
declare -a WANTED=("$@")
selected() {
  local name="$1"
  [ ${#WANTED[@]} -eq 0 ] && return 0
  local w
  for w in "${WANTED[@]}"; do
    [ "$w" = "$name" ] && return 0
  done
  return 1
}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: interpreter '$PYTHON' not found (set PYTHON=... to override)" >&2
  exit 1
fi

total=0
passed=0
failed=0
started=$(date +%s)

printf 'Seed loop: %s\n' "$SEEDS"
printf 'Logs:      %s\n\n' "$OUT_DIR"

for entry in "${METHODS_ALL[@]}"; do
  IFS='|' read -r name dir script <<< "$entry"
  selected "$name" || continue

  if [ ! -f "$ROOT/$dir/$script" ]; then
    echo "[skip] $name — $dir/$script not found"
    continue
  fi

  for seed in $SEEDS; do
    total=$((total + 1))
    log="$OUT_DIR/${name//+/_}_seed${seed}.log"
    printf '[%2d] %-22s seed=%-4s ... ' "$total" "$name" "$seed"

    t0=$(date +%s)
    (
      cd "$ROOT/$dir" || exit 1
      "$PYTHON" "$script" --seed "$seed"
    ) > "$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))

    # Pull the compact summary out of the log (quiet mode => stdout is just that)
    {
      echo "### $name seed=$seed  (rc=$rc, ${dt}s)"
      grep -E '^(Model:|batch=|accuracy:|precision:|recall:|f1:|val_loss:)' "$log"
      echo
    } >> "$SUMMARY_FILE"

    if [ $rc -eq 0 ]; then
      passed=$((passed + 1))
      echo "ok  (${dt}s)  ->  ${log#$ROOT/}"
    else
      failed=$((failed + 1))
      echo "FAILED rc=$rc  (${dt}s)  ->  ${log#$ROOT/}"
      echo "$name seed=$seed rc=$rc log=${log#$ROOT/}" >> "$FAIL_FILE"
      # keep the tail of the log visible on the terminal for quick diagnosis
      tail -n 15 "$log" | sed 's/^/      | /'
    fi
  done
done

elapsed=$(( $(date +%s) - started ))

printf '\n%s\n' '------------------------------------------------------------'
printf 'Done in %dm%02ds — total=%d  ok=%d  failed=%d\n' \
       $((elapsed / 60)) $((elapsed % 60)) "$total" "$passed" "$failed"
printf 'Summaries: %s\n' "${SUMMARY_FILE#$ROOT/}"
if [ "$failed" -gt 0 ]; then
  printf 'Failures:  %s\n' "${FAIL_FILE#$ROOT/}"
  cat "$FAIL_FILE"
fi
printf '%s\n' '------------------------------------------------------------'

[ "$failed" -eq 0 ] || exit 1
