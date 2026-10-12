#!/usr/bin/env bash
# Supply CPU route/resources to sbatch. KEY=VALUE ... -- watcher arguments.
set -euo pipefail
while [ "$#" -gt 0 ] && [ "$1" != -- ]; do export "$1"; shift; done
[ "${1:-}" = -- ] || { echo 'Expected -- before watcher arguments' >&2; exit 2; }
shift
export PATH="${UV_BIN_DIR:?}:/usr/bin:/bin:${PATH:-}"
cd "${REPO:?}"
exec uv run --python "${WATCH_ENV:?}/bin/python" --no-project --offline \
    python slurm/watch-smoke.py "$@"
