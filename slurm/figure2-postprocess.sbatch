#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PILOT_ENTRYPOINT=figure2-postprocess.sbatch
source "$REPO/slurm/shorthand-runtime.sh"
export PATH="${UV_BIN_DIR:?}:$PATH"
case "${PHASE:?}" in
    select) ARGS=(eval/select_figure2.py --root "$RUN_ROOT" --out "$RUN_ROOT/models.json");;
    plot) ARGS=(eval/figure2.py --results "$RESULTS_ROOT" --out "$RESULTS_ROOT/figure2");;
    *) exit 2;;
esac
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python "${ARGS[@]}"
