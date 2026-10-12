#!/usr/bin/env bash
# CPU setup: PROFILE=inference|training|grading, TANDEM_ENV, RUN_ROOT and site paths.
# Installs environments only; dataset preparation and downloads are separate entry points.
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
: "${PROFILE:?}" "${TANDEM_ENV:?}" "${RUN_ROOT:?}"
case "$PROFILE" in inference|training|grading) ;; *) echo 'Unknown setup profile' >&2; exit 2;; esac
export PATH="${UV_BIN_DIR:?}:${PATH:-/usr/bin:/bin}"
mkdir -p "$RUN_ROOT"
if [ "$PROFILE" = training ] && [ "${PREPARE_FORKS:-0}" = 1 ] && [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ]; then
    bash third_party/apply_patches.sh
fi
if [ -n "${APPTAINER_IMAGE:-}" ] && [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ]; then
    APPTAINER="${APPTAINER_BIN_DIR:-/orcd/software/core/001/pkg/apptainer/1.4.2/bin}/apptainer"
    if [ ! -f "$APPTAINER_IMAGE" ]; then
        export APPTAINER_CACHEDIR="$RUN_ROOT/apptainer-cache"
        export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/tandem-image-${SLURM_JOB_ID:-$$}"
        mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$APPTAINER_IMAGE")"
        "$APPTAINER" build --mksquashfs-args "-processors ${SLURM_CPUS_PER_TASK:-4} -mem 2048M" \
            "$APPTAINER_IMAGE" \
            docker://nvidia/cuda@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7
    fi
    export TANDEM_CONTAINER_ACTIVE=1
    exec "$APPTAINER" exec --bind "${CONTAINER_BIND:?}" "$APPTAINER_IMAGE" bash "$REPO/slurm/setup.sh"
fi
if [ "${TANDEM_CONTAINER_ACTIVE:-0}" = 1 ]; then
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
    # Only CPU setup links against CUDA's stub. GPU jobs use --nv and real drivers.
    mkdir -p "$RUN_ROOT/cuda-stubs"
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "$RUN_ROOT/cuda-stubs/libcuda.so.1"
    export LD_LIBRARY_PATH="$RUN_ROOT/cuda-stubs:${LD_LIBRARY_PATH:-}"
fi
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/tandem-setup-${SLURM_JOB_ID:-$$}"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
mkdir -p "$XDG_CACHE_HOME"
[ -x "$TANDEM_ENV/bin/python" ] || uv venv --python 3.11 "$TANDEM_ENV"
run_python() { uv run --python "$TANDEM_ENV/bin/python" --no-project python "$@"; }
# Bypass the damaged shared ANTLR cache without modifying other users' entries.
uv pip install --python "$TANDEM_ENV/bin/python" --no-cache antlr4-python3-runtime==4.9.3
REQUIREMENTS=env/figure2-requirements.txt
if [ "$PROFILE" = grading ]; then REQUIREMENTS=env/grading-requirements.txt; fi
uv pip install --python "$TANDEM_ENV/bin/python" -r "$REQUIREMENTS" -c env/requirements-freeze.txt
if [ "$PROFILE" = training ]; then
    uv pip install --python "$TANDEM_ENV/bin/python" setuptools wheel setuptools-scm
    SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0.dev0 uv pip install --python "$TANDEM_ENV/bin/python" \
        -e third_party/verl -c env/requirements-freeze.txt -c env/constraints.txt TransferQueue==0.1.8
    WHEEL_URL=$(run_python - <<'PY'
import json, urllib.request
with urllib.request.urlopen('https://pypi.org/pypi/vllm/0.19.1/json') as response:
    wheels = json.load(response)['urls']
print(next(w['url'] for w in wheels if 'x86_64' in w['filename'] and w['filename'].endswith('.whl')))
PY
)
    SETUPTOOLS_SCM_PRETEND_VERSION=0.19.1 VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL_URL" \
        uv pip install --python "$TANDEM_ENV/bin/python" -e third_party/vllm --no-build-isolation --no-deps
    run_python - <<'PY'
from vllm import _C
from vllm.config import TandemConfig
from vllm.v1.sample.tandem_sampler import TandemSampler
from verl.workers.utils.losses import _apply_tandem_senior_gate
print('Patched training imports passed')
PY
    run_python -m unittest discover -s tests -p test_senior_gate.py
    run_python -m unittest discover -s tests -p test_word_state_machine.py
    run_python -m unittest discover -s tests -p test_validation_mode.py
fi
run_python -m unittest discover -s tests -p test_math_grading.py
run_python -m unittest discover -s tests -p test_benchmark_aggregation.py
if [ "$PROFILE" != grading ]; then
    run_python -c 'from vllm import LLM; from transformers import AutoTokenizer'
    run_python -m unittest discover -s tests -p test_figure2.py
fi
uv pip freeze --python "$TANDEM_ENV/bin/python" > "$RUN_ROOT/$PROFILE-environment.txt"
echo "$PROFILE SETUP COMPLETE"
