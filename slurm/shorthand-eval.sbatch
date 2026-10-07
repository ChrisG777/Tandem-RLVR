#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
[ "${SLURM_RESTART_COUNT:-0}" -le 3 ] || exit 1
export PATH="/data/scratch/chrisge/.local/bin:/usr/bin:/bin"
export HF_HOME=/data/vision/torralba/u/chrisge/.cache/huggingface
export UV_CACHE_DIR=/data/vision/torralba/u/chrisge/.cache/uv
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export XDG_CACHE_HOME="/tmp/shorthand-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
mkdir -p "$XDG_CACHE_HOME"
unset VLLM_TANDEM_CONFIG VLLM_TANDEM_ALL_GPUS
run_python() { uv run --python "${TANDEM_ENV:?}/bin/python" --no-project --offline python "$@"; }
export BASE_MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"]["path"])' "$RUN_ROOT/base.json")
TASKS=(manipulate_matrix string_manipulation)
DATA=()
case "${PHASE:?}" in
  calibration)
    for TASK in "${TASKS[@]}"; do DATA+=("$DATA_ROOT/$TASK/calibration.parquet"); done
    MODEL="$BASE_MODEL"; OUTPUT="$RUN_ROOT/calibration";;
  base)
    for TASK in "${TASKS[@]}"; do
      DATA+=("$DATA_ROOT/$TASK/test.parquet" "$DATA_ROOT/$TASK/long.parquet")
    done
    MODEL="$BASE_MODEL"; OUTPUT="$RUN_ROOT/eval/base";;
  trained)
    TASK=${TASKS[$((SLURM_ARRAY_TASK_ID / 2))]}
    BUDGETS=(256 1024); BUDGET=${BUDGETS[$((SLURM_ARRAY_TASK_ID % 2))]}
    NAME="$TASK-b$BUDGET"
    test -f "$RUN_ROOT/train/$NAME/verified.json"
    MODEL="$RUN_ROOT/train/$NAME/hf/global_step_100"
    OUTPUT="$RUN_ROOT/eval/$NAME"
    DATA=("$DATA_ROOT/$TASK/test.parquet" "$DATA_ROOT/$TASK/long.parquet");;
  *) exit 2;;
esac
run_python eval/shorthand.py --model "$MODEL" --data "${DATA[@]}" --out-dir "$OUTPUT" --n 4
if [ "$PHASE" = calibration ]; then
  run_python train/check_shorthand.py --calibration "$RUN_ROOT/calibration" --data-root "$DATA_ROOT"
fi
