#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PILOT_ENTRYPOINT=shorthand-train.sbatch
source "$REPO/slurm/shorthand-runtime.sh"
[ "${SLURM_RESTART_COUNT:-0}" -le 3 ] || exit 1
export PATH="${UV_BIN_DIR:-/data/scratch/chrisge/.local/bin}:${PATH:-/usr/bin:/bin}"
export TANDEM_ENV_BIN="${TANDEM_ENV:?}/bin" TANDEM_ENV_FILE="$REPO/slurm/training-env.sh"
export HF_HOME=${HF_HOME:-/data/vision/torralba/u/chrisge/.cache/huggingface}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/data/vision/torralba/u/chrisge/.cache/uv}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
export XDG_CACHE_HOME="/tmp/shorthand-train-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export RAY_TMPDIR="/tmp/sr${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME" "$RAY_TMPDIR"
run_python() { uv run --python "$TANDEM_ENV_BIN/python" --no-project --offline python "$@"; }
if [ "${CALIBRATION_VERIFIED:-0}" = 1 ]; then
    test -f "$RUN_ROOT/calibration/verified.json"
else
    run_python train/check_shorthand.py --calibration "$RUN_ROOT/calibration" --data-root "$DATA_ROOT" --tasks ${PILOT_TASKS:-manipulate_matrix string_manipulation} --budgets ${PILOT_BUDGETS:-256 1024}
fi
export BASE_MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"]["path"])' "$RUN_ROOT/base.json")
read -ra TASKS <<< "${PILOT_TASKS:-manipulate_matrix string_manipulation}"; read -ra BUDGETS <<< "${PILOT_BUDGETS:-256 1024}"
export TASK=${TASKS[$((SLURM_ARRAY_TASK_ID / 2))]}
export MAX_RESPONSE=${BUDGETS[$((SLURM_ARRAY_TASK_ID % 2))]}
export EXP_NAME="$TASK-b$MAX_RESPONSE" PROJECT_NAME=tandem-shorthand-solo-pilot
export CKPT_ROOT="$RUN_ROOT/train" CKPT_DIR="$RUN_ROOT/train/$EXP_NAME"
export TRAIN_FILES="$DATA_ROOT/$TASK/train.parquet"
export MAX_PROMPT=1536 TRAIN_BATCH=16 MINI_BATCH=8 ROLLOUT_N=8 ROLLOUT_TEMP=0.6 LR=1e-6
export TOTAL_STEPS=${TOTAL_STEPS:-100} SAVE_FREQ=${SAVE_FREQ:-10} TEST_FREQ=${TEST_FREQ:-25} VAL_N=4 VAL_TEMP=0.6 VAL_TOP_P=1 VAL_TOP_K=-1
export MAX_CKPT_KEEP=1 PPO_MAX_TOKEN=4096
mkdir -p "$CKPT_DIR"
export VERL_FILE_LOGGER_PATH="$CKPT_DIR/metrics-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.jsonl"
echo "$CODE_COMMIT" > "$CKPT_DIR/code-${SLURM_JOB_ID}.txt"
cp "$DATA_ROOT/$TASK/manifest.json" "$CKPT_DIR/data-manifest.json"
cp "$RUN_ROOT/base.json" "$CKPT_DIR/initialization.json"
nvidia-smi > "$CKPT_DIR/gpus-${SLURM_JOB_ID}.txt"
bash train/vanilla_grpo.sh \
  "data.val_files=[$DATA_ROOT/$TASK/validation.parquet]" \
  "reward.custom_reward_function.path=$REPO/reward/shorthand_reward.py" \
  'trainer.logger=[console,file]' data.seed=42 \
  "ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK" \
  actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.top_p=1 \
  actor_rollout_ref.rollout.top_k=-1 trainer.resume_mode=auto \
  "hydra.run.dir=$CKPT_DIR/hydra-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}"
if [ "${DEFER_TRAINING_CHECK:-0}" != 1 ]; then
    run_python train/check_shorthand.py --training "$CKPT_DIR" --steps "$TOTAL_STEPS" --task "$TASK"
fi
