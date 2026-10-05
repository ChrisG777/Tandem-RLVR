#!/usr/bin/env bash
# Existing training entry points; site options supplied to sbatch.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/train-%j.out
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
[ "${SLURM_RESTART_COUNT:-0}" -le 3 ] || { echo 'Preemption retry limit reached'; exit 1; }
export PATH="/data/scratch/${SLURM_JOB_USER:-chrisge}/.local/bin:/usr/bin:/bin"
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
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1
mkdir -p "$XDG_CACHE_HOME" "$RAY_TMPDIR"
run_python() { uv run --python "$TANDEM_ENV_BIN/python" --no-project --offline python "$@"; }
export BASE_MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"]["path"])' "$RUN_ROOT/base.json")
export POLICY_MODEL="$BASE_MODEL" JUNIOR_MODEL="$BASE_MODEL"
export TOTAL_STEPS=200 TEST_FREQ=20 SAVE_FREQ=20
EXTRA=()
case "$MODE" in
    smoke) export TOTAL_STEPS=3 TEST_FREQ=3 SAVE_FREQ=3; EXTRA+=(data.val_max_samples=8);;
    full) ;;
    *) echo "MODE must be smoke or full" >&2; exit 2;;
esac
case "$ARM" in
    tandem) LAUNCHER=train/tandem_grpo.sh;;
    grpo) LAUNCHER=train/vanilla_grpo.sh;;
    *) echo "ARM must be tandem or grpo" >&2; exit 2;;
esac
git rev-parse HEAD > "$CKPT_DIR/code-${SLURM_JOB_ID}.txt"
cp "$RUN_ROOT/base.json" "$CKPT_DIR/initialization.json"
cp "$DATA_ROOT/deepscaler/manifest.json" "$CKPT_DIR/data-manifest.json"
nvidia-smi > "$CKPT_DIR/gpus-${SLURM_JOB_ID}.txt"
# Same data order and rollout seeds for both arms. Resume is only within this run.
bash "$LAUNCHER" 'trainer.logger=[console,file]' data.seed=42 \
    actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.top_k=-1 trainer.resume_mode=auto \
    "hydra.run.dir=$CKPT_DIR/hydra-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}" "${EXTRA[@]}"
run_python train/figure2_checkpoint.py --root "$CKPT_DIR" --arm "$ARM" --steps "$TOTAL_STEPS" \
    --mode "$MODE"
