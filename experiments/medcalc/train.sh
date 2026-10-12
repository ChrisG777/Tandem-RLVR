#!/usr/bin/env bash
# Train one MedCalc arm: GRPO (senior alone) or tandem (senior + frozen junior).
#
#   bash experiments/medcalc/train.sh grpo smoke
#   bash experiments/medcalc/train.sh grpo full
#   bash experiments/medcalc/train.sh tandem full
#
# Re-running the same command resumes from that arm's checkpoint (trainer.resume_mode=auto).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENV_FILE=${TANDEM_ENV_FILE:-$ROOT/experiments/medcalc/env.sh}
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE. Run bash experiments/medcalc/install_a100.sh first." >&2
  exit 1
fi
# The launchers in train/ default to train/env.sh unless this is set.
export TANDEM_ENV_FILE="$ENV_FILE"
# shellcheck disable=SC1090
source "$ENV_FILE"
cd "$ROOT"

ARM=${1:?usage: train.sh grpo|tandem smoke|full}
MODE=${2:-full}
case "$ARM" in
  grpo|tandem) ;;
  *) echo "arm must be grpo or tandem" >&2; exit 2 ;;
esac
case "$MODE" in
  smoke|full) ;;
  *) echo "mode must be smoke or full" >&2; exit 2 ;;
esac

DATA=${MEDCALC_DATA:-$ROOT/data/medcalc}
BASE_JSON=$DATA/base.json
BOUNDARY=$DATA/qwen3_word_boundary_ids.json
if [[ ! -f "$BASE_JSON" || ! -f "$DATA/train.parquet" || ! -f "$BOUNDARY" ]]; then
  echo "missing prepared data under $DATA. Run bash experiments/medcalc/prepare.sh first." >&2
  exit 1
fi

PY=${TANDEM_ENV_BIN:?}/python
# shellcheck disable=SC1091
source "$ROOT/experiments/medcalc/ensure_python_dev.sh"
ensure_python_dev "$PY"
eval "$("$PY" - <<'PY'
import shlex
from experiments.medcalc.protocol import shell_defaults
for key, value in shell_defaults().items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

export BASE_MODEL=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$BASE_JSON")
export POLICY_MODEL=$BASE_MODEL
export JUNIOR_MODEL=$BASE_MODEL
export BOUNDARY_IDS_PATH=$BOUNDARY
export TRAIN_FILES=$DATA/train.parquet
export VAL_FILES="[$DATA/heldout.parquet]"
export REWARD_FN=$ROOT/experiments/medcalc/reward.py
export TRAIN_LOGGER='[console,file]'
export PROJECT_NAME=${PROJECT_NAME:-medcalc-trlvr}
export CKPT_DIR=$CKPT_ROOT/medcalc/${ARM}-${MODE}
export EXP_NAME=medcalc_${ARM}_${MODE}
mkdir -p "$CKPT_DIR"
export VERL_FILE_LOGGER_PATH=$CKPT_DIR/metrics-$(date +%s)-0.jsonl
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
# vLLM's CuMemAllocator refuses PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *expandable_segments:True* ]]; then
  echo "removing expandable_segments:True from PYTORCH_CUDA_ALLOC_CONF; vLLM's memory pool rejects it" >&2
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF//expandable_segments:True/}
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF//,,/,}
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF#,}
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF%,}
  if [[ -z "$PYTORCH_CUDA_ALLOC_CONF" ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
  else
    export PYTORCH_CUDA_ALLOC_CONF
  fi
fi

if [[ "$MODE" == "smoke" ]]; then
  export TOTAL_STEPS=${MEDCALC_STEPS:-$SMOKE_STEPS}
  export TEST_FREQ=$TOTAL_STEPS
  export SAVE_FREQ=$TOTAL_STEPS
  export PERSIST_HF_INTERVAL=$TOTAL_STEPS
else
  export TOTAL_STEPS=${MEDCALC_STEPS:-$TRAIN_STEPS}
  export TEST_FREQ=${TEST_FREQ:-20}
  export SAVE_FREQ=${SAVE_FREQ:-$TEST_FREQ}
  export PERSIST_HF_INTERVAL=${PERSIST_HF_INTERVAL:-$SAVE_FREQ}
fi
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
if (( TOTAL_STEPS % TEST_FREQ != 0 )); then
  echo "TOTAL_STEPS=$TOTAL_STEPS must be a multiple of TEST_FREQ=$TEST_FREQ." >&2
  echo "Selection reads the held-out score at the final step. Use MEDCALC_STEPS=200 or set TEST_FREQ to a divisor." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Train on the A100s, not this machine." >&2
  exit 1
fi
mapfile -t MEMS < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')
NGPU=${#MEMS[@]}
MEM=${MEMS[0]:-0}
echo "visible GPUs: $NGPU, memory: ${MEMS[*]} MiB"

if [[ "$ARM" == "tandem" ]]; then
  if [[ "$NGPU" -lt 2 ]]; then
    echo "Tandem needs two 80 GB A100s: senior on GPU 0, frozen junior on GPU 1." >&2
    echo "This machine has $NGPU visible GPU(s)." >&2
    exit 1
  fi
  for gpu_mem in "${MEMS[@]:0:2}"; do
    if [[ "$gpu_mem" -lt 70000 ]]; then
      echo "Tandem needs two 80 GB A100s. A visible GPU has ${gpu_mem} MiB." >&2
      exit 1
    fi
  done
  RAM_GIB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
  if [[ "$RAM_GIB" -lt 120 ]]; then
    echo "warning: host RAM is ${RAM_GIB} GiB. FSDP offloads the optimizer; use at least 128 GiB." >&2
  fi
fi
export TRAIN_BATCH=${TRAIN_BATCH:-16}
export MINI_BATCH=${MINI_BATCH:-8}
export ROLLOUT_N=${ROLLOUT_N:-8}
export MAX_PROMPT MAX_RESPONSE
export PPO_MAX_TOKEN=${PPO_MAX_TOKEN:-$(( MAX_PROMPT + MAX_RESPONSE + 256 ))}

if [[ "$ARM" == "grpo" ]]; then
  export GPUS=${GPUS:-0}
  export TRAIN_GPUS=${TRAIN_GPUS:-1}
  export ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL:-0.50}
  LAUNCHER=$ROOT/train/vanilla_grpo.sh
else
  export GPUS=${GPUS:-0,1}
  export FROZEN_GPU=${FROZEN_GPU:-1}
  export TRAIN_GPUS=${TRAIN_GPUS:-1}
  export ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL:-0.50}
  echo "tandem layout: senior on GPU 0, frozen junior on GPU 1"
  LAUNCHER=$ROOT/train/tandem_grpo.sh
fi

EXTRA=()
if [[ "$MODE" == "smoke" ]]; then
  EXTRA+=(data.val_max_samples=${MEDCALC_VAL_SAMPLES:-8})
elif [[ -n "${MEDCALC_VAL_SAMPLES:-}" ]]; then
  EXTRA+=(data.val_max_samples="$MEDCALC_VAL_SAMPLES")
fi

echo "arm=$ARM mode=$MODE steps=$TOTAL_STEPS batch=$TRAIN_BATCH rollouts=$ROLLOUT_N util=$ROLLOUT_GPU_UTIL"
echo "checkpoints: $CKPT_DIR"
echo "metrics: $VERL_FILE_LOGGER_PATH"

bash "$LAUNCHER" \
  data.seed=42 \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.top_p=1 \
  actor_rollout_ref.rollout.top_k=-1 \
  trainer.resume_mode=auto \
  "+data.apply_chat_template_kwargs.enable_thinking=False" \
  "hydra.run.dir=$CKPT_DIR/hydra-$(date +%s)" \
  "${EXTRA[@]}"

if ! "$PY" "$ROOT/train/figure2_checkpoint.py" \
  --root "$CKPT_DIR" \
  --arm "$ARM" \
  --steps "$TOTAL_STEPS" \
  --mode "$MODE" \
  --validation-metric "$VALIDATION_METRIC"
then
  echo "Checkpoint selection failed. Metric keys containing 'val' or 'acc' in the latest log:" >&2
  "$PY" - "$CKPT_DIR" <<'PY' || true
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = sorted(root.glob("metrics-*.jsonl"))
if not files:
    print("no metrics JSONL found", file=sys.stderr)
    raise SystemExit
row = json.loads(files[-1].read_text().splitlines()[-1])
keys = sorted(k for k in row.get("data", {}) if "val" in k or "acc" in k)
print("\n".join(keys) if keys else "(no val/acc keys in the last row)", file=sys.stderr)
PY
  exit 1
fi

echo "selected checkpoint: $CKPT_DIR/selected.json"
if [[ "$ARM" == "tandem" ]]; then
  echo "confirm actor/tandem_senior_token_frac in the metrics JSONL is near 0.5."
  echo "absent means the junior mask never reached the loss and the run is plain GRPO."
fi
