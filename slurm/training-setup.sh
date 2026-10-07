#!/usr/bin/env bash
# CPU allocation. Forks must already be cloned and patched at their pinned commits.
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PATH="${UV_BIN_DIR:-/data/scratch/${SLURM_JOB_USER:-chrisge}/.local/bin}:${PATH:-/usr/bin:/bin}"
if [ "${PREPARE_FORKS:-0}" = 1 ] && [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ]; then
    bash third_party/apply_patches.sh
fi
if [ -n "${APPTAINER_IMAGE:-}" ] && [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ]; then
    export TANDEM_CONTAINER_ACTIVE=1
    exec "${APPTAINER_BIN_DIR:-/orcd/software/core/001/pkg/apptainer/1.4.2/bin}/apptainer" exec \
        --bind "${CONTAINER_BIND:?}" "$APPTAINER_IMAGE" bash "$REPO/slurm/training-setup.sh"
fi
if [ "${TANDEM_CONTAINER_ACTIVE:-0}" = 1 ]; then export CC=/usr/bin/gcc CXX=/usr/bin/g++; fi
export UV_LINK_MODE=hardlink
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/data/scratch/${SLURM_JOB_USER:-chrisge}/.local/share/uv/python}"
export XDG_CACHE_HOME="/tmp/tandem-setup-${SLURM_JOB_ID}"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
mkdir -p "$XDG_CACHE_HOME" "$RUN_ROOT"
[ -x "$TANDEM_ENV/bin/python" ] || uv venv --python 3.11 "$TANDEM_ENV"
run_python() { uv run --python "$TANDEM_ENV/bin/python" --no-project python "$@"; }
# Avoid two damaged entries in the site's shared cache.
uv pip install --python "$TANDEM_ENV/bin/python" --no-cache antlr4-python3-runtime==4.9.3 sympy==1.14.0
uv pip install --python "$TANDEM_ENV/bin/python" -r env/figure2-requirements.txt \
    -c env/requirements-freeze.txt setuptools wheel setuptools-scm
SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0.dev0 uv pip install --python "$TANDEM_ENV/bin/python" -e third_party/verl \
    -c env/requirements-freeze.txt -c env/constraints.txt TransferQueue==0.1.8
WHEEL_URL=$(run_python - <<'PY'
import json, urllib.request
with urllib.request.urlopen('https://pypi.org/pypi/vllm/0.19.1/json') as response:
    wheels = json.load(response)['urls']
print(next(w['url'] for w in wheels if 'x86_64' in w['filename'] and w['filename'].endswith('.whl')))
PY
)
SETUPTOOLS_SCM_PRETEND_VERSION=0.19.1 VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL_URL" \
    uv pip install --python "$TANDEM_ENV/bin/python" -e third_party/vllm \
    --no-build-isolation --no-deps
run_python - <<'PY'
import torch, vllm, verl, transfer_queue
from vllm import _C
from vllm.config import TandemConfig
from vllm.v1.sample.tandem_sampler import TandemSampler
from verl.workers.utils.losses import _apply_tandem_senior_gate
print('Patched training imports passed:', torch.__version__, vllm.__version__)
PY
run_python -m unittest discover -s tests
uv pip freeze --python "$TANDEM_ENV/bin/python" > "$RUN_ROOT/training-environment.txt"
run_python eval/download_figure2.py --out "$RUN_ROOT/base.json"
if [ "${PREPARE_DEEPSCALER:-1}" = 1 ] && [ ! -f "$DATA_ROOT/deepscaler/manifest.json" ]; then
    run_python data/build_deepscaler.py --out-dir "$DATA_ROOT/deepscaler"
fi
echo 'TRAINING SETUP COMPLETE'
