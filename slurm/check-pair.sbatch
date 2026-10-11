#!/usr/bin/env bash
# Native backend diagnostic; site route supplied by sbatch.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export JOB_ENTRYPOINT=slurm/check-pair.sbatch
export RESUME_PROGRESS_FILE="${RUN_ROOT:?}/pair-probe.json"
source slurm/runtime.sh
export PATH="${UV_BIN_DIR:?}:$PATH"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="/tmp/qwen35-probe-${SLURM_JOB_ID}"
export XDG_CONFIG_HOME="$XDG_CACHE_HOME/config" VLLM_CONFIG_ROOT="$XDG_CACHE_HOME/config/vllm"
export TRITON_CACHE_DIR="$RUN_ROOT/triton-cache"
mkdir -p "$XDG_CACHE_HOME"
[ "${SLURM_RESTART_COUNT:-0}" -le "${MAX_RESTARTS:-12}" ] || exit 1
nvidia-smi
uv run --no-project --offline --python "${TANDEM_ENV:?}/bin/python" python train/check_pair.py \
    --models "$RUN_ROOT/base.json" --boundaries "$RUN_ROOT/word-boundaries.json" \
    --out "$RUN_ROOT/pair-probe.json" &
wait "$!"
