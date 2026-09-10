#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$EVAL_DIR/.." && pwd)

if [ -f "$ROOT/train/env.sh" ]; then
    source "$ROOT/train/env.sh"
fi
if [ -n "${TANDEM_ENV_BIN:-}" ]; then
    export PATH="$TANDEM_ENV_BIN:$PATH"
fi
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
export TOKENIZERS_PARALLELISM=false

: "${MODEL:?set MODEL to the senior checkpoint}"
: "${TAG:?set TAG to a name for this checkpoint, for example step_120 or base}"
JUNIOR=${JUNIOR:-${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}}
RESULTS_ROOT=${RESULTS_ROOT:-$ROOT/results}
N=${N:-8}
LIMIT=${LIMIT:-0}

BENCH_ARGS=()
if [ -n "${BENCHMARKS:-}" ]; then BENCH_ARGS=(--benchmarks "$BENCHMARKS"); fi
GPU_ARGS=()
if [ -n "${GPU_UTIL:-}" ]; then GPU_ARGS=(--gpu-util "$GPU_UTIL"); fi
HANDOFF_ARGS=()
if [ -n "${SINGLE_GPU:-}" ]; then HANDOFF_ARGS=(--single-gpu); fi

OUT=$RESULTS_ROOT/$TAG
mkdir -p "$OUT"

phase() {
    local name=$1
    shift
    if [ -f "$OUT/$name.json" ]; then
        echo "skip $name ($TAG)"
        return
    fi
    echo "=== $name ($TAG) ==="
    python "$@"
}

phase solo "$EVAL_DIR/solo.py" \
    --model "$MODEL" --out "$OUT/solo.json" \
    --n "$N" --limit "$LIMIT" "${BENCH_ARGS[@]}" "${GPU_ARGS[@]}"

phase handoff "$EVAL_DIR/handoff.py" \
    --senior "$MODEL" --junior "$JUNIOR" --out "$OUT/handoff.json" \
    --n "$N" --limit "$LIMIT" "${BENCH_ARGS[@]}" "${GPU_ARGS[@]}" "${HANDOFF_ARGS[@]}"

phase legibility "$EVAL_DIR/legibility.py" \
    --solo "$OUT/solo.json" --junior "$JUNIOR" --out "$OUT/legibility.json"

echo "=== done $TAG, results in $OUT ==="
