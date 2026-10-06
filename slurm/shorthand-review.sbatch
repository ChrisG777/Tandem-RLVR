#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PATH="/data/scratch/chrisge/.local/bin:/usr/bin:/bin"
export UV_CACHE_DIR=/data/vision/torralba/u/chrisge/.cache/uv
uv run --python "${TANDEM_ENV:?}/bin/python" --no-project --offline \
  python eval/review_shorthand.py --run-root "${RUN_ROOT:?}"
