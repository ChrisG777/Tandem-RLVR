#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PILOT_ENTRYPOINT=shorthand-eval.sbatch
source "$REPO/slurm/shorthand-runtime.sh"
[ "${SLURM_RESTART_COUNT:-0}" -le "${MAX_RESTARTS:-12}" ] || exit 1
export PATH="${UV_BIN_DIR:-/data/scratch/chrisge/.local/bin}:${PATH:-/usr/bin:/bin}"
export HF_HOME=${HF_HOME:-/data/vision/torralba/u/chrisge/.cache/huggingface}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/data/vision/torralba/u/chrisge/.cache/uv}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export XDG_CACHE_HOME="/tmp/shorthand-${SLURM_JOB_ID}"
export XDG_CONFIG_HOME="$XDG_CACHE_HOME/config"
export VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CONFIG_ROOT"
unset VLLM_TANDEM_CONFIG VLLM_TANDEM_ALL_GPUS
run_python() { uv run --python "${TANDEM_ENV:?}/bin/python" --no-project --offline python "$@"; }
read -ra BUDGETS <<< "${PILOT_BUDGETS:-256 1024}"
export BASE_MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"]["path"])' "$RUN_ROOT/base.json")
read -ra TASKS <<< "${PILOT_TASKS:-manipulate_matrix string_manipulation}"
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
    read -ra BUDGETS <<< "${PILOT_BUDGETS:-256 1024}"; BUDGET=${BUDGETS[$((SLURM_ARRAY_TASK_ID % 2))]}
    NAME="$TASK-b$BUDGET"
    test -f "$RUN_ROOT/train/$NAME/verified.json"
    MODEL="$RUN_ROOT/train/$NAME/hf/global_step_100"
    OUTPUT="$RUN_ROOT/eval/$NAME"
    DATA=("$DATA_ROOT/$TASK/test.parquet" "$DATA_ROOT/$TASK/long.parquet");;
  *) exit 2;;
esac
run_python eval/shorthand.py --model "$MODEL" --data "${DATA[@]}" --out-dir "$OUTPUT" --n 4 --budgets "${BUDGETS[@]}" --max-model-len "${PILOT_CONTEXT:-4096}" &
wait $!
if [ "$PHASE" = calibration ]; then
  run_python train/check_shorthand.py --calibration "$RUN_ROOT/calibration" --data-root "$DATA_ROOT" --tasks ${PILOT_TASKS:-manipulate_matrix string_manipulation} --budgets ${PILOT_BUDGETS:-256 1024}
fi
