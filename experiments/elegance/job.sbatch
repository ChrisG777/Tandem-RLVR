#!/usr/bin/env bash
# Resource requests and paths are supplied explicitly at submission.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PATH="${UV_BIN_DIR:?}:$PATH"
export JOB_ENTRYPOINT=experiments/elegance/job.sbatch
export MAX_RESTARTS=60
export CKPT_DIR="${RUN_ROOT:?}/${ARM:?}"
export RESUME_CHECKPOINT_ROOT="$CKPT_DIR"
source slurm/runtime.sh
[ "${SLURM_RESTART_COUNT:-0}" -le 60 ] || exit 1
export TANDEM_ENV_BIN="${TANDEM_ENV:?}/bin" TANDEM_ENV_FILE="$REPO/slurm/training-env.sh"
export CKPT_ROOT="$RUN_ROOT" PROJECT_NAME=method-elegance EXP_NAME="elegance-$ARM"
export XDG_CACHE_HOME="/tmp/elegance-${SLURM_JOB_ID}" RAY_TMPDIR="/tmp/er${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TILELANG_CACHE_DIR="$XDG_CACHE_HOME/tilelang"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1
export TRAIN_GPUS=1 FROZEN_GPU=0 OMP_NUM_THREADS=4
export ELEGANCE_JUDGE_CACHE="$RUN_ROOT/judge-cache"
export ELEGANCE_CODEX_BIN=/data/scratch/chrisge/.local/bin/codex-cluster
export TOTAL_STEPS=${TOTAL_STEPS:-200} TOTAL_EPOCHS=100 TEST_FREQ=20 SAVE_FREQ=5
export MAX_CKPT_KEEP=2 PERSIST_HF_INTERVAL=200 PPO_MAX_TOKEN=4096
export MAX_PROMPT=2048 MAX_RESPONSE=3000
export BOUNDARY_IDS_PATH="${INITIALIZATION_ROOT:?}/word-boundaries.json"
export TRAIN_FILES="${ELEGANCE_DATA:?}/train.parquet"
export VERL_FILE_LOGGER_PATH="$CKPT_DIR/metrics-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.jsonl"
mkdir -p "$CKPT_DIR" "$XDG_CACHE_HOME" "$RAY_TMPDIR"
run_python() { uv run --python "$TANDEM_ENV_BIN/python" --no-project --offline python "$@"; }
run_python -c 'import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); owned=sum(f.stat().st_blocks*512 for f in p.rglob("*") if f.is_file()); assert shutil.disk_usage(p).free + owned > 280 * 1024**3, "Insufficient capacity for overlapping full-weight resume checkpoints"' "$CKPT_DIR"
run_python -m experiments.elegance.preflight --archive data/method-elegance/dataset.zip \
  --cache "$ELEGANCE_JUDGE_CACHE" --codex "$ELEGANCE_CODEX_BIN" \
  --receipt "$CKPT_DIR/judge-preflight-${SLURM_JOB_ID}.json"
export BASE_MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"]["path"])' "$INITIALIZATION_ROOT/base.json")
export POLICY_MODEL="$BASE_MODEL"
export JUNIOR_MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["junior"]["path"])' "$INITIALIZATION_ROOT/base.json")
cp "$INITIALIZATION_ROOT/base.json" "$CKPT_DIR/initialization.json"
cp "$ELEGANCE_DATA/manifest.json" "$CKPT_DIR/data-manifest.json"
git rev-parse HEAD > "$CKPT_DIR/code-${SLURM_JOB_ID}.txt"
run_python -c 'import importlib.metadata as m,json,os; print(json.dumps({"packages":{n:m.version(n) for n in ("torch","transformers","vllm","verl","math-verify")},"pythonpath":os.environ.get("PYTHONPATH")},indent=2))' > "$CKPT_DIR/runtime-${SLURM_JOB_ID}.json"
nvidia-smi > "$CKPT_DIR/gpus-${SLURM_JOB_ID}.txt"
case "$ARM" in
  tandem) LAUNCHER=train/tandem_grpo.sh;;
  grpo) LAUNCHER=train/vanilla_grpo.sh;;
  *) exit 2;;
esac
# Both arms share this exact objective and data; no held-out test is loaded.
bash "$LAUNCHER" 'trainer.logger=[console,file]' data.seed=42 \
  "data.val_files=[$ELEGANCE_DATA/validation.parquet]" \
  "reward.custom_reward_function.path=$REPO/experiments/elegance/reward.py" \
  data.filter_overlong_prompts=False \
  "ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK}" \
  actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.top_p=1 \
  actor_rollout_ref.rollout.top_k=-1 trainer.resume_mode=auto \
  '+data.apply_chat_template_kwargs.enable_thinking=False' \
  actor_rollout_ref.actor.freeze_vision_tower=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
  '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:0,video:0}' \
  "hydra.run.dir=$CKPT_DIR/hydra-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}" &
TRAIN_PID=$!
wait "$TRAIN_PID"
