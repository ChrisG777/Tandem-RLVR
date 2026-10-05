#!/usr/bin/env bash
set -euo pipefail
export "$@"
export PATH=/usr/bin:/bin
cd "$REPO"
exec /data/scratch/chrisge/.local/bin/uv run --python "$WATCH_ENV/bin/python" --no-project --offline \
    python slurm/watch_figure2.py --smoke-job "$SMOKE" --full-job "$FULL" \
    --results "$RUN_ROOT/$ARM-smoke" --status "$RUN_ROOT/$ARM-smoke-status.json" --training-arm "$ARM"
