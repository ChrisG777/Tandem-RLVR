#!/usr/bin/env bash
# Existing training entry points; site options supplied to sbatch.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --nice=0
#SBATCH --output=logs/train-%j.out
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PILOT_ENTRYPOINT=train-figure2.sbatch
source "$REPO/slurm/shorthand-runtime.sh"
[ "${SLURM_RESTART_COUNT:-0}" -le "${MAX_RESTARTS:-12}" ] || { echo 'Preemption retry limit reached'; exit 1; }
export PATH="${UV_BIN_DIR:-/data/scratch/${SLURM_JOB_USER:-chrisge}/.local/bin}:$PATH"
export TANDEM_ENV_BIN="${TANDEM_ENV:?}/bin"
export TANDEM_ENV_FILE="$REPO/slurm/training-env.sh"
export CKPT_ROOT="${RUN_ROOT:?}" PROJECT_NAME=tandem-figure2-fresh
export CKPT_DIR="$RUN_ROOT/${ARM:?}-${MODE:?}"
export EXP_NAME="$ARM-$MODE"
mkdir -p "$CKPT_DIR"
export VERL_FILE_LOGGER_PATH="$CKPT_DIR/metrics-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.jsonl"
export XDG_CACHE_HOME="/tmp/tandem-train-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TILELANG_CACHE_DIR="$XDG_CACHE_HOME/tilelang"
if [ -n "${CUDA_HOME:-}" ]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1
mkdir -p "$XDG_CACHE_HOME" "$RAY_TMPDIR"
run_python() { uv run --python "$TANDEM_ENV_BIN/python" --no-project --offline python "$@"; }
export BASE_MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"]["path"])' "$RUN_ROOT/base.json")
export POLICY_MODEL="$BASE_MODEL"
export JUNIOR_MODEL=$(run_python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("junior",d["base"])["path"])' "$RUN_ROOT/base.json")
if [ -f "$RUN_ROOT/word-boundaries.json" ]; then export BOUNDARY_IDS_PATH="$RUN_ROOT/word-boundaries.json"; fi
export TOTAL_STEPS=200 TEST_FREQ=20 SAVE_FREQ=${SAVE_FREQ:-20}
export PPO_MAX_TOKEN=${PPO_MAX_TOKEN:-4096}
export PERSIST_HF_INTERVAL=20
EXTRA=()
if [ -f "$RUN_ROOT/word-boundaries.json" ]; then
    export FROZEN_GPU=${FROZEN_GPU:-0}
    EXTRA+=('+data.apply_chat_template_kwargs.enable_thinking=False'
        +trainer.require_complete_validation=True
        actor_rollout_ref.actor.freeze_vision_tower=True
        actor_rollout_ref.actor.strategy=fsdp2
        actor_rollout_ref.actor.fsdp_config.offload_policy=True
        actor_rollout_ref.model.use_remove_padding=False
        actor_rollout_ref.rollout.enable_prefix_caching=False
        +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:0,video:0}')
fi
if [ "${QWEN35_SPEED:-0}" = 1 ]; then
    test -f "$RUN_ROOT/word-boundaries.json" || { echo 'Qwen speed mode requires a pair manifest'; exit 2; }
    source "$REPO/train/execution.sh"
    export TRAIN_PROFILE_DIR=${TRAIN_PROFILE_DIR:-${QWEN35_PROFILE_DIR:-}}
    configure_training_execution qwen35 optimized
    EXTRA+=("${EXECUTION_EXTRA[@]}"
        "trainer.rollout_data_dir=$CKPT_DIR/rollouts"
        "trainer.validation_data_dir=$CKPT_DIR/validation")
fi
case "$MODE" in
    smoke)
        export TOTAL_STEPS=${SMOKE_STEPS:-3}
        export TEST_FREQ=$TOTAL_STEPS SAVE_FREQ=$TOTAL_STEPS PERSIST_HF_INTERVAL=$TOTAL_STEPS
        EXTRA+=("data.val_max_samples=${SMOKE_VAL_SAMPLES:-8}");;
    full) ;;
    *) echo "MODE must be smoke or full" >&2; exit 2;;
esac
case "$ARM" in
    tandem) LAUNCHER=train/tandem_grpo.sh;;
    grpo) LAUNCHER=train/vanilla_grpo.sh;;
    *) echo "ARM must be tandem or grpo" >&2; exit 2;;
esac
echo "$CODE_COMMIT" > "$CKPT_DIR/code-${SLURM_JOB_ID}.txt"
cp "$RUN_ROOT/base.json" "$CKPT_DIR/initialization.json"
cp "$DATA_ROOT/deepscaler/manifest.json" "$CKPT_DIR/data-manifest.json"
run_python - <<'PY' > "$CKPT_DIR/runtime-${SLURM_JOB_ID}.json"
import importlib.metadata as metadata
import json
import os
import sys
names = sorted({distribution.metadata["Name"] for distribution in metadata.distributions()})
print(json.dumps({"python": sys.version, "pythonpath": os.environ.get("PYTHONPATH", ""),
                  "cuda_home": os.environ.get("CUDA_HOME", ""),
                  "packages": {name: metadata.version(name) for name in names}}, indent=2))
PY
nvidia-smi > "$CKPT_DIR/gpus-${SLURM_JOB_ID}.txt"
# Same data order and rollout seeds for both arms. Resume is only within this run.
bash "$LAUNCHER" 'trainer.logger=[console,file]' data.seed=42 \
    "ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK}" \
    actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.top_k=-1 trainer.resume_mode=auto \
    "hydra.run.dir=$CKPT_DIR/hydra-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}" "${EXTRA[@]}" &
TRAIN_PID=$!
wait "$TRAIN_PID"
VERIFY_ARGS=()
if [ -n "${EXPECTED_RESUME_STEP:-}" ]; then
    VERIFY_ARGS+=(--expected-resume-step "$EXPECTED_RESUME_STEP" --resume-log "${RESUME_LOG:?}")
fi
run_python train/figure2_checkpoint.py --root "$CKPT_DIR" --arm "$ARM" --steps "$TOTAL_STEPS" \
    --mode "$MODE" "${VERIFY_ARGS[@]}"
