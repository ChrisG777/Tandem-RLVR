#!/usr/bin/env bash
# CPU-only grading; supply the site's CPU account/partition/QoS at submission.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --nice=0
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PATH="${UV_BIN_DIR:?}:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ "${MODE:-grade}" = setup ]; then
    [ -x "${GRADING_ENV:?}/bin/python" ] || uv venv --python 3.11 "$GRADING_ENV"
    uv pip install --python "$GRADING_ENV/bin/python" -r env/grading-requirements.txt
    uv run --no-project --python "$GRADING_ENV/bin/python" python -m unittest discover -s tests -p test_math_grading.py
    exit
fi
uv run --no-project --offline --python "${GRADING_ENV:?}/bin/python" python eval/regrade.py \
    --input "${SOURCE:?}" --out "${OUTPUT:?}" --cache "${GRADE_CACHE:?}" \
    --workers "${SLURM_CPUS_PER_TASK:-4}"
