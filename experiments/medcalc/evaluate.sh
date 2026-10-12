#!/usr/bin/env bash
# Evaluate the base model and the selected GRPO / Tandem seniors.
#
#   bash experiments/medcalc/evaluate.sh          # paper sample counts
#   bash experiments/medcalc/evaluate.sh check    # smoke checkpoints, 8 problems
#
# Solo is the senior alone. Handoff alternates with the frozen base junior at
# blank-line reasoning steps. Legibility is the junior's per-token cross-entropy
# on the senior's solo traces.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENV_FILE=${TANDEM_ENV_FILE:-$ROOT/experiments/medcalc/env.sh}
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE. Run bash experiments/medcalc/install_a100.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"
cd "$ROOT"
PY=${TANDEM_ENV_BIN:?}/python

KIND=${1:-full}
case "$KIND" in
  full)
    TRAIN_MODE=${MEDCALC_TRAIN_MODE:-full}
    LIMIT=${MEDCALC_LIMIT:-0}
    SOLO_N=${MEDCALC_SOLO_N:-32}
    HANDOFF_N=${MEDCALC_HANDOFF_N:-8}
    ;;
  check)
    TRAIN_MODE=${MEDCALC_TRAIN_MODE:-smoke}
    LIMIT=${MEDCALC_LIMIT:-8}
    SOLO_N=${MEDCALC_SOLO_N:-2}
    HANDOFF_N=${MEDCALC_HANDOFF_N:-2}
    ;;
  *)
    echo "usage: evaluate.sh [full|check]" >&2
    exit 2
    ;;
esac

DATA=${MEDCALC_DATA:-$ROOT/data/medcalc}
BASE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$DATA/base.json")
BENCHMARKS=$("$PY" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["eval_benchmarks"]))' "$DATA/manifest.json")
CONTEXT=$("$PY" -c 'from experiments.medcalc.protocol import CONTEXT, MAX_PROMPT, MAX_RESPONSE; print(CONTEXT)')
MAX_PROMPT=$("$PY" -c 'from experiments.medcalc.protocol import MAX_PROMPT; print(MAX_PROMPT)')
MAX_RESPONSE=$("$PY" -c 'from experiments.medcalc.protocol import MAX_RESPONSE; print(MAX_RESPONSE)')
OUT=${MEDCALC_RESULTS:-$ROOT/results/medcalc/$TRAIN_MODE}
mkdir -p "$OUT"

model_for() {
  local arm=$1
  if [[ "$arm" == "base" ]]; then
    printf '%s\n' "$BASE"
    return
  fi
  local selected=$CKPT_ROOT/medcalc/${arm}-${TRAIN_MODE}/selected.json
  if [[ ! -f "$selected" ]]; then
    echo "missing $selected. Finish bash experiments/medcalc/train.sh $arm $TRAIN_MODE first." >&2
    exit 1
  fi
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$selected"
}

run_solo() {
  local arm=$1 model=$2
  mkdir -p "$OUT/$arm"
  "$PY" "$ROOT/eval/solo.py" \
    --model "$model" \
    --out "$OUT/$arm/solo.json" \
    --n "$SOLO_N" \
    --limit "$LIMIT" \
    --benchmarks "$BENCHMARKS" \
    --data-root "$DATA/eval" \
    --context "$CONTEXT" \
    --max-prompt "$MAX_PROMPT" \
    --max-response "$MAX_RESPONSE"
}

run_handoff() {
  local arm=$1 model=$2
  local util=()
  if [[ -n "${MEDCALC_GPU_UTIL:-}" ]]; then
    util=(--gpu-util "$MEDCALC_GPU_UTIL")
  fi
  "$PY" "$ROOT/eval/handoff.py" \
    --senior "$model" \
    --junior "$BASE" \
    --out "$OUT/$arm/handoff.json" \
    --n "$HANDOFF_N" \
    --limit "$LIMIT" \
    --benchmarks "$BENCHMARKS" \
    --data-root "$DATA/eval" \
    --context "$CONTEXT" \
    --max-prompt "$MAX_PROMPT" \
    --max-response "$MAX_RESPONSE" \
    "${util[@]}"
}

run_legibility() {
  local arm=$1
  "$PY" "$ROOT/eval/legibility.py" \
    --solo "$OUT/$arm/solo.json" \
    --junior "$BASE" \
    --out "$OUT/$arm/legibility.json"
}

echo "results: $OUT"
echo "benchmarks: $BENCHMARKS"
echo "samples: solo=$SOLO_N handoff=$HANDOFF_N limit=$LIMIT"

for arm in base grpo tandem; do
  model=$(model_for "$arm")
  echo "=== $arm solo: $model"
  run_solo "$arm" "$model"
  if [[ "$arm" != "base" ]]; then
    echo "=== $arm handoff with frozen junior"
    run_handoff "$arm" "$model"
  fi
  if [[ "${MEDCALC_SKIP_LEGIBILITY:-0}" != "1" ]]; then
    echo "=== $arm junior cross-entropy"
    run_legibility "$arm"
  fi
done

"$PY" "$ROOT/experiments/medcalc/report.py" \
  --results "$OUT" \
  --panel "$DATA/panel.json" \
  --out "$OUT/summary"
