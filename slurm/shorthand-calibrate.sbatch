#!/usr/bin/env bash
# Site-neutral, two-task calibration array; paths/account/partition supplied at submission.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
: "${RUN_ROOT:?}" "${DATA_ROOT:?}" "${TANDEM_ENV:?}" "${HF_HOME:?}" "${UV_CACHE_DIR:?}"
export PATH="${UV_BIN_DIR:-${HOME:?}/.local/bin}:${PATH:-/usr/bin:/bin}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export XDG_CACHE_HOME="/tmp/shorthand-calibrate-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
mkdir -p "$XDG_CACHE_HOME" "$RUN_ROOT/calibration"
unset VLLM_TANDEM_CONFIG VLLM_TANDEM_ALL_GPUS
run_python() { uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python "$@"; }
TASKS=(manipulate_matrix string_manipulation)
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:?}]}
MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"]["path"])' "$RUN_ROOT/base.json")
git rev-parse HEAD > "$RUN_ROOT/calibration/code-${SLURM_JOB_ID}.txt"
nvidia-smi > "$RUN_ROOT/calibration/gpus-${SLURM_JOB_ID}.txt"
# These probe larger budgets after the original 256/1024 pilot failed its gate.
# They do not certify training eligibility or select a budget using test results.
run_python eval/shorthand.py --model "$MODEL" --data "$DATA_ROOT/$TASK/calibration.parquet" \
  --out-dir "$RUN_ROOT/calibration" --budgets 2048 3072 --n 4
