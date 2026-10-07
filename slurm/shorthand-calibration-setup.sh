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
    # CUDA devel supplies the C/CUDA toolchain needed by Triton's runtime JIT.
    # Official NVIDIA CUDA 12.8.1 devel / Ubuntu 22.04, pinned amd64 manifest.
    apptainer build --mksquashfs-args "-processors ${SLURM_CPUS_PER_TASK:-4} -mem 2048M" \
      "$APPTAINER_IMAGE" \
      docker://nvidia/cuda@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7
  fi
  export TANDEM_CONTAINER_ACTIVE=1
  exec apptainer exec --bind "$CONTAINER_BIND" "$APPTAINER_IMAGE" \
    bash "$REPO/slurm/shorthand-calibration-setup.sh"
fi
if [ "${TANDEM_CONTAINER_ACTIVE:-0}" = 1 ]; then
  export CC=/usr/bin/gcc CXX=/usr/bin/g++
  printf 'int main(void){return 0;}\n' | "$CC" -x c - -o "/tmp/tandem-cc-${SLURM_JOB_ID}"
  "/tmp/tandem-cc-${SLURM_JOB_ID}"
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
