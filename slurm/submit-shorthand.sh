#!/usr/bin/env bash
# Submit the four solo pilots and their evaluations after successful calibration.
set -euo pipefail
cd "${REPO:?}"
: "${RUN_ROOT:?}" "${DATA_ROOT:?}" "${TANDEM_ENV:?}" "${CALIBRATION_JOB:?}"
[ ! -e "$RUN_ROOT/training-submitted.txt" ] || { echo 'Campaign already submitted'; exit 1; }
SITE=("REPO=$REPO" "RUN_ROOT=$RUN_ROOT" "DATA_ROOT=$DATA_ROOT" "TANDEM_ENV=$TANDEM_ENV")
COMMON=(--account=vision-torralba --qos=shared-if-available --export=NIL
        --chdir="$REPO" --kill-on-invalid-dep=yes)
TRAIN=$(sbatch --parsable "${COMMON[@]}" --array=0-3 \
    --partition=vision-shared-h200,vision-shared-h100,vision-shared-a100 \
    '--constraint=nvidia_h200|nvidia_h100_80gb_hbm3|nvidia_h100_nvl|nvidia_a100-sxm4-80gb' \
    --dependency="afterok:$CALIBRATION_JOB" --job-name=shorthand-solo \
    --output="$REPO/logs/shorthand-train-%A_%a.out" slurm/shorthand-train.sbatch "${SITE[@]}")
printf 'train %s\n' "$TRAIN" | tee "$RUN_ROOT/training-submitted.txt" >> "$RUN_ROOT/jobs.txt"
EVAL_POOLS=vision-shared-h200,vision-shared-h100,vision-shared-a100,vision-shared-l40s,vision-shared-a6000,vision-shared-rtx6000ada
BASE=$(sbatch --parsable "${COMMON[@]}" --partition="$EVAL_POOLS" \
    --dependency="afterok:$CALIBRATION_JOB" --job-name=shorthand-base \
    --output="$REPO/logs/shorthand-base-%j.out" slurm/shorthand-eval.sbatch "${SITE[@]}" PHASE=base)
printf 'base %s\n' "$BASE" >> "$RUN_ROOT/jobs.txt"
EVAL=$(sbatch --parsable "${COMMON[@]}" --partition="$EVAL_POOLS" --array=0-3 \
    --dependency="aftercorr:$TRAIN" --job-name=shorthand-eval \
    --output="$REPO/logs/shorthand-eval-%A_%a.out" slurm/shorthand-eval.sbatch "${SITE[@]}" PHASE=trained)
printf 'eval %s\n' "$EVAL" >> "$RUN_ROOT/jobs.txt"
REVIEW=$(sbatch --parsable --partition=tig-cpu --account=csail --qos=tig-main \
    --cpus-per-task=1 --mem=2G --time=00:15:00 --export=NIL --chdir="$REPO" \
    --kill-on-invalid-dep=yes --dependency="afterok:$BASE:$EVAL" --job-name=shorthand-review \
    --output="$REPO/logs/shorthand-review-%j.out" slurm/shorthand-review.sbatch "${SITE[@]}")
printf 'review %s\n' "$REVIEW" >> "$RUN_ROOT/jobs.txt"
cat "$RUN_ROOT/jobs.txt"
