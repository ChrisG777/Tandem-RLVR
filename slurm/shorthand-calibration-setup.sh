#!/usr/bin/env bash
# CPU-only environment and model setup for the independent calibration race.
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
: "${RUN_ROOT:?}" "${TANDEM_ENV:?}" "${HF_HOME:?}" "${UV_CACHE_DIR:?}"
if [ -n "${APPTAINER_IMAGE:-}" ] && [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ]; then
  # Login nodes have /usr/bin/apptainer; some compute nodes need the shared module.
  export PATH="${APPTAINER_BIN_DIR:-/orcd/software/core/001/pkg/apptainer/1.4.2/bin}:${PATH:-/usr/bin:/bin}"
  : "${CONTAINER_BIND:?}"
  export APPTAINER_CACHEDIR="$RUN_ROOT/apptainer-cache"
  export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/shorthand-image-${SLURM_JOB_ID}"
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$APPTAINER_IMAGE")"
  if [ ! -f "$APPTAINER_IMAGE" ]; then
    # Official NVIDIA CUDA 12.8.1 runtime / Ubuntu 22.04, pinned amd64 manifest.
    apptainer build --mksquashfs-args "-processors ${SLURM_CPUS_PER_TASK:-4} -mem 2048M" \
      "$APPTAINER_IMAGE" \
      docker://nvidia/cuda@sha256:fcbbd60a5ad3db3a1c7375bf14546b369b54064c513224310b2026df50c7a9bd
  fi
  export TANDEM_CONTAINER_ACTIVE=1
  exec apptainer exec --bind "$CONTAINER_BIND" "$APPTAINER_IMAGE" \
    bash "$REPO/slurm/shorthand-calibration-setup.sh"
fi
export PATH="${UV_BIN_DIR:-${HOME:?}/.local/bin}:${PATH:-/usr/bin:/bin}"
mkdir -p "$RUN_ROOT" "$HF_HOME" "$UV_CACHE_DIR"
[ -x "$TANDEM_ENV/bin/python" ] || uv venv --python 3.11 "$TANDEM_ENV"
getconf GNU_LIBC_VERSION
uv pip install --only-binary vllm --python "$TANDEM_ENV/bin/python" -r env/figure2-requirements.txt \
  -c env/requirements-freeze.txt
uv pip freeze --python "$TANDEM_ENV/bin/python" > "$RUN_ROOT/calibration-environment.txt"
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python -c \
  'import platform, torch; from importlib.metadata import version; print("Runtime:", platform.libc_ver(), torch.__version__, version("vllm")); assert version("vllm") == "0.19.1"'
uv run --python "$TANDEM_ENV/bin/python" --no-project python eval/download_figure2.py \
  --out "$RUN_ROOT/base.json"
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python \
  -m unittest discover -s tests -p test_shorthand.py -v
