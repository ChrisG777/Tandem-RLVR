#!/usr/bin/env bash
# CPU-only integrity and calibration gate; depend on GPU calibration and setup.
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PILOT_ENTRYPOINT=shorthand-check.sbatch
source "$REPO/slurm/shorthand-runtime.sh"
export PATH="${UV_BIN_DIR:?}:$PATH"
export HF_HUB_OFFLINE=1
ARGS=(--calibration "$RUN_ROOT/calibration" --data-root "$DATA_ROOT"
      --tasks ${PILOT_TASKS:-manipulate_matrix string_manipulation} --budgets ${PILOT_BUDGETS:-256 1024})
if [ "${PHASE:-calibration}" = training ]; then
    read -ra TASKS <<< "${PILOT_TASKS:-manipulate_matrix string_manipulation}"
    read -ra BUDGETS <<< "${PILOT_BUDGETS:-256 1024}"
    TASK=${TASKS[$((SLURM_ARRAY_TASK_ID / 2))]}
    BUDGET=${BUDGETS[$((SLURM_ARRAY_TASK_ID % 2))]}
    ARGS=(--training "$RUN_ROOT/train/$TASK-b$BUDGET" --task "$TASK" --steps 100)
fi
uv run --python "${TANDEM_ENV:?}/bin/python" --no-project --offline python \
    train/check_shorthand.py "${ARGS[@]}"
