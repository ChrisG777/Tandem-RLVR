#!/usr/bin/env bash
# Bounded qualification in one existing interactive H200 allocation.
# Writes per-case logs/exit receipts under PROFILE_ROOT, then releases via salloc.
# Independent cases continue after a case failure; shared kernel failure stops all.
set -euo pipefail
export REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${PROFILE_ROOT:?}" "${SLURM_JOB_ID:?}"
mkdir -p "$PROFILE_ROOT/checks"
cd "$REPO"
exec > >(tee -a "$PROFILE_ROOT/suite-${SLURM_JOB_ID}.log") 2>&1
export UV_BIN_DIR=/data/scratch/chrisge/.local/bin
export PATH="$UV_BIN_DIR:$PATH"
export TANDEM_ENV=/data/vision/torralba/u/chrisge/tandem-rlvr/train-venv
export INITIALIZATION_ROOT=/data/vision/torralba/u/chrisge/tandem-rlvr/qwen35-9b-2b
export CUDA_HOME="$INITIALIZATION_ROOT/cuda-12.8.1"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$REPO/tests:$REPO/third_party/verl:$REPO:$INITIALIZATION_ROOT/training-kernels:$REPO/third_party/vllm:$TANDEM_ENV/lib/python3.11/site-packages:/data/vision/torralba/u/chrisge/tandem-rlvr/regrading-20261009/venv/lib/python3.11/site-packages"
export HF_HOME=/data/vision/torralba/u/chrisge/.cache/huggingface
export UV_CACHE_DIR=/data/vision/torralba/u/chrisge/.cache/uv
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="/tmp/shared-rl-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TILELANG_CACHE_DIR="$XDG_CACHE_HOME/tilelang"
mkdir -p "$XDG_CACHE_HOME"
run_python() { uv run --no-project --offline --python "$TANDEM_ENV/bin/python" python "$@"; }
git rev-parse HEAD > "$PROFILE_ROOT/code-${SLURM_JOB_ID}.txt"
nvidia-smi > "$PROFILE_ROOT/checks/gpu-${SLURM_JOB_ID}.txt"
run_python -c 'import torch; assert torch.cuda.is_available(); assert "H200" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name())'
run_python -m unittest test_qwen35_kernels test_fsdp2_offload_accumulation -v > "$PROFILE_ROOT/checks/kernels-${SLURM_JOB_ID}.log" 2>&1
printf '0\n' > "$PROFILE_ROOT/checks/kernels-${SLURM_JOB_ID}.exit"
failed=0
for kind in apps elegance; do
    for arm in grpo tandem; do
        case_root="$PROFILE_ROOT/$kind/$arm"
        mkdir -p "$case_root"
        # Skip only a previously successful three-step qualification.
        if [ -f "$case_root/profile.exit" ] && [ "$(cat "$case_root/profile.exit")" = 0 ]; then continue; fi
        echo "Starting $kind $arm profiling at $(date -u +%FT%TZ)"
        if bash train/profile-experiment.sh "$kind" "$arm" > "$case_root/profile-${SLURM_JOB_ID}.log" 2>&1; then
            printf '0\n' > "$case_root/profile.exit"
        else
            code=$?
            printf '%s\n' "$code" > "$case_root/profile.exit"
            failed=1
        fi
        echo "Finished $kind $arm: $(cat "$case_root/profile.exit")"
    done
done
printf '%s\n' "$failed" > "$PROFILE_ROOT/suite.exit"
exit "$failed"
