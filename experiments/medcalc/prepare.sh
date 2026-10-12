#!/usr/bin/env bash
# Download the pinned Qwen3-4B base, build word-boundary ids, and write MedCalc parquet.
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
OUT=${MEDCALC_DATA:-$ROOT/data/medcalc}

"$PY" "$ROOT/experiments/medcalc/build_data.py" --out "$OUT" --download-base "$@"
BASE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$OUT/base.json")
"$PY" "$ROOT/train/assets/build_word_boundary_ids.py" \
  --model "$BASE" \
  --out "$OUT/qwen3_word_boundary_ids.json" \
  --write \
  --expected-count "$("$PY" -c 'from experiments.medcalc.protocol import BOUNDARY_COUNT; print(BOUNDARY_COUNT)')"

echo "prepared $OUT"
echo "next, on the A100:"
echo "  bash experiments/medcalc/train.sh grpo smoke"
echo "  bash experiments/medcalc/train.sh tandem smoke"
