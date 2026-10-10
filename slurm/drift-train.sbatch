#!/usr/bin/env bash
# Route/resources must be supplied after checking the live allocation policy.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PILOT_ENTRYPOINT=drift-train.sbatch
export CKPT_DIR="${RUN_ROOT:?}/${MODEL_KEY:?}-lr${LR:-1e-5}-s${SEED:-42}"
source "$REPO/slurm/shorthand-runtime.sh"
[ "${SLURM_RESTART_COUNT:-0}" -le 12 ] || exit 1
export PATH="${UV_BIN_DIR:?}:$PATH"
export TANDEM_ENV_BIN="${TANDEM_ENV:?}/bin" TANDEM_ENV_FILE="$REPO/slurm/training-env.sh"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
export XDG_CACHE_HOME="/tmp/drift-${SLURM_JOB_ID}" RAY_TMPDIR="/tmp/dr${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
mkdir -p "$CKPT_DIR" "$XDG_CACHE_HOME" "$RAY_TMPDIR"
run_python() { uv run --no-project --offline --python "$TANDEM_ENV_BIN/python" python "$@"; }
export BASE_MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$DATA_ROOT/$MODEL_KEY.json")
TEMPLATE=$(run_python -c 'import json; from data.build_drift import RAW_TEMPLATE; print(json.dumps(RAW_TEMPLATE))')
export TRAIN_FILES="$DATA_ROOT/train.parquet" MAX_PROMPT=1024 MAX_RESPONSE=256
export TRAIN_BATCH=128 MINI_BATCH=128 ROLLOUT_N=64 ROLLOUT_TEMP=0.5 LR=${LR:-1e-5}
export TOTAL_EPOCHS=2 TOTAL_STEPS=${TOTAL_STEPS:-114} TEST_FREQ=10 SAVE_FREQ=10
export VAL_N=1 VAL_DO_SAMPLE=False VAL_TEMP=0 VAL_TOP_P=1 VAL_TOP_K=-1
export PPO_MAX_TOKEN=2048 ROLLOUT_GPU_UTIL=0.55 MAX_CKPT_KEEP=2 PERSIST_HF_INTERVAL=10
export EXP_NAME=$(basename "$CKPT_DIR") PROJECT_NAME=natural-language-drift CKPT_ROOT="$RUN_ROOT"
export VERL_FILE_LOGGER_PATH="$CKPT_DIR/metrics-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.jsonl"
cp "$DATA_ROOT/manifest.json" "$CKPT_DIR/data-manifest.json"
cp "$DATA_ROOT/$MODEL_KEY.json" "$CKPT_DIR/initialization.json"
printf '%s\n' "$CODE_COMMIT" > "$CKPT_DIR/code-${SLURM_JOB_ID}.txt"
EXTRA_ARGS=()
if [ "${CONFIG_ONLY:-0}" = 1 ]; then
    EXTRA_ARGS=(--cfg job)
else
    nvidia-smi > "$CKPT_DIR/gpus-${SLURM_JOB_ID}.txt"
fi
bash train/vanilla_grpo.sh \
  "data.val_files=[$DATA_ROOT/validation.parquet]" "data.seed=${SEED:-42}" \
  "actor_rollout_ref.model.custom_chat_template=$TEMPLATE" \
  "reward.custom_reward_function.path=$REPO/reward/gsm8k_drift.py" \
  'trainer.logger=[console,file]' "ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK" \
  "actor_rollout_ref.rollout.seed=${SEED:-42}" \
  actor_rollout_ref.rollout.top_p=1 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  trainer.resume_mode=auto trainer.val_before_train=True \
  "trainer.validation_data_dir=$CKPT_DIR/validation" \
  "trainer.rollout_data_dir=$CKPT_DIR/rollouts" \
  "hydra.run.dir=$CKPT_DIR/hydra-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}" "${EXTRA_ARGS[@]}" &
wait $!
