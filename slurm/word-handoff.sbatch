#!/usr/bin/env bash
# Site route supplied at submission; both decoders share one GPU.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --nice=0
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PILOT_ENTRYPOINT=word-handoff.sbatch
source "$REPO/slurm/shorthand-runtime.sh"
[ "${SLURM_RESTART_COUNT:-0}" -le "${MAX_RESTARTS:-12}" ] || exit 1
export PATH="${UV_BIN_DIR:?}:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/tmp/word-handoff-${SLURM_JOB_ID}"
export XDG_CONFIG_HOME="$XDG_CACHE_HOME/config" VLLM_CONFIG_ROOT="$XDG_CACHE_HOME/config/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
mkdir -p "$XDG_CACHE_HOME" "$(dirname "${OUTPUT:?}")"
uv run --python "${TANDEM_ENV:?}/bin/python" --no-project --offline python eval/word_handoff.py \
    --senior "${SENIOR:?}" --junior "${JUNIOR:?}" --out "$OUTPUT" \
    --n "${N:-8}" --limit "${LIMIT:-0}" --decoding "${DECODING:-training}" &
wait "$!"
