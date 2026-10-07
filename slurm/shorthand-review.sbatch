#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PILOT_ENTRYPOINT=shorthand-review.sbatch
source "$REPO/slurm/shorthand-runtime.sh"
export PATH="${UV_BIN_DIR:-/data/scratch/chrisge/.local/bin}:${PATH:-/usr/bin:/bin}"
export UV_CACHE_DIR=${UV_CACHE_DIR:-/data/vision/torralba/u/chrisge/.cache/uv}
uv run --python "${TANDEM_ENV:?}/bin/python" --no-project --offline \
  python eval/review_shorthand.py --run-root "${RUN_ROOT:?}" --expected-comparisons "${EXPECTED_COMPARISONS:-16}"
