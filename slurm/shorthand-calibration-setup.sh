#!/usr/bin/env bash
# CPU-only environment and model setup for the independent calibration race.
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
: "${RUN_ROOT:?}" "${TANDEM_ENV:?}" "${HF_HOME:?}" "${UV_CACHE_DIR:?}"
export PATH="${UV_BIN_DIR:-${HOME:?}/.local/bin}:${PATH:-/usr/bin:/bin}"
mkdir -p "$RUN_ROOT" "$HF_HOME" "$UV_CACHE_DIR"
[ -x "$TANDEM_ENV/bin/python" ] || uv venv --python 3.11 "$TANDEM_ENV"
uv pip install --python "$TANDEM_ENV/bin/python" -r env/figure2-requirements.txt \
  -c env/requirements-freeze.txt
uv pip freeze --python "$TANDEM_ENV/bin/python" > "$RUN_ROOT/calibration-environment.txt"
uv run --python "$TANDEM_ENV/bin/python" --no-project python eval/download_figure2.py \
  --out "$RUN_ROOT/base.json"
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python \
  -m unittest discover -s tests -p test_shorthand.py -v
