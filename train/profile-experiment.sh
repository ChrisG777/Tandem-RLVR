#!/usr/bin/env bash
# profile-experiment.sh apps|elegance grpo|tandem
# Run three disposable full-weight steps plus full validation in an existing
# single-GPU allocation. Inputs are the explicit cluster paths below; outputs
# go to PROFILE_ROOT/KIND/ARM. Never resumes or mutates a production checkpoint.
set -euo pipefail
KIND=${1:?apps or elegance}; export ARM=${2:?grpo or tandem}
case "$KIND:$ARM" in apps:grpo|apps:tandem|elegance:grpo|elegance:tandem) ;; *) exit 2;; esac
: "${SLURM_JOB_ID:?Existing interactive allocation required}"
export REPO=$(cd "$(dirname "$0")/.." && pwd)
export RUN_ROOT="${PROFILE_ROOT:?}/$KIND/$ARM"
export UV_BIN_DIR=${UV_BIN_DIR:-/data/scratch/chrisge/.local/bin}
export PATH="$UV_BIN_DIR:$PATH"
export TANDEM_ENV=${TANDEM_ENV:-/data/vision/torralba/u/chrisge/tandem-rlvr/train-venv}
export INITIALIZATION_ROOT=${INITIALIZATION_ROOT:-/data/vision/torralba/u/chrisge/tandem-rlvr/qwen35-9b-2b}
export CUDA_HOME="$INITIALIZATION_ROOT/cuda-12.8.1"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME=/data/vision/torralba/u/chrisge/.cache/huggingface
export UV_CACHE_DIR=/data/vision/torralba/u/chrisge/.cache/uv
export PYTHONPATH="$REPO/third_party/verl:$REPO:$INITIALIZATION_ROOT/training-kernels:$REPO/third_party/vllm:$TANDEM_ENV/lib/python3.11/site-packages:/data/vision/torralba/u/chrisge/tandem-rlvr/regrading-20261009/venv/lib/python3.11/site-packages"
export MODE=profile PROFILE_STEPS=3 PROFILE_TEST_FREQ=3 TRAIN_EXECUTION=optimized
unset TRAIN_PROFILE_DIR QWEN35_PROFILE_DIR VLLM_TANDEM_CONFIG VLLM_TANDEM_ALL_GPUS VERL_FILE_LOGGER_PATH
cd "$REPO"
if [ "$KIND" = apps ]; then
    export DATA_ROOT=/data/vision/torralba/u/chrisge/tandem-rlvr/apps-20261010-6k/data
    export ROLLOUT_MAX_SEQS=32 ROLLOUT_KV_BYTES=51539607552
    if [ "$ARM" = grpo ]; then export ROLLOUT_MAX_SEQS=64 ROLLOUT_KV_BYTES=103079215104; fi
    bash experiments/coding/job.sbatch train
else
    export DATA_ROOT=/data/vision/torralba/u/chrisge/tandem-rlvr/data-fresh-20261005
    export ELEGANCE_DATA=/data/vision/torralba/u/chrisge/tandem-rlvr/elegance-20261011/data
    export ELEGANCE_CODEX_BIN="$REPO/experiments/elegance/codex-runtime.py"
    export ELEGANCE_CODEX_HOME=/data/vision/torralba/u/chrisge/.codex-elegance-profile
    bash experiments/elegance/job.sbatch
fi
