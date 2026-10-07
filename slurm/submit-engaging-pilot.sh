#!/usr/bin/env bash
# Matrix-only continuation of the calibrated pilot. No CSAIL mutations.
set -euo pipefail
cd "${REPO:?}"
: "${RUN_ROOT:?}" "${DATA_ROOT:?}" "${TANDEM_ENV:?}" "${EVAL_ENV:?}"
: "${SETUP_JOB:?}" "${CALIBRATION_JOB:?}" "${TRAIN_EXCLUDE:?}" "${EVAL_EXCLUDE:?}"
[ ! -e "$RUN_ROOT/training-submitted.txt" ] || { echo 'Campaign already submitted'; exit 1; }
SITE=("REPO=$REPO" "RUN_ROOT=$RUN_ROOT" "DATA_ROOT=$DATA_ROOT"
      "HF_HOME=${HF_HOME:?}" "UV_CACHE_DIR=${UV_CACHE_DIR:?}" "UV_BIN_DIR=${UV_BIN_DIR:?}"
      "APPTAINER_IMAGE=${APPTAINER_IMAGE:?}" "CONTAINER_BIND=${CONTAINER_BIND:?}"
      PILOT_TASKS=manipulate_matrix 'PILOT_BUDGETS=2048 3072')
COMMON=(--parsable --account=mit_general --qos=normal --export=NIL
        --chdir="$REPO" --kill-on-invalid-dep=yes)
CPU=("${COMMON[@]}" --partition=mit_normal --cpus-per-task=1 --mem=4G --time=01:00:00)
GPU=("${COMMON[@]}" --partition=mit_normal_gpu,mit_preemptable --gpus=1 --requeue)
GATE=$(sbatch "${CPU[@]}" --dependency="afterok:$SETUP_JOB:$CALIBRATION_JOB" \
    --job-name=pilot-data-check --output="$REPO/logs/pilot-data-check-%j.out" \
    slurm/shorthand-check.sbatch "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV")
TRAIN=$(sbatch "${GPU[@]}" --array=0-1 --exclude="$TRAIN_EXCLUDE" \
    --cpus-per-task=6 --mem=144G --time=06:00:00 --dependency="afterok:$GATE" \
    --job-name=pilot-solo --output="$REPO/logs/pilot-solo-%A_%a.out" \
    slurm/shorthand-train.sbatch "${SITE[@]}" "TANDEM_ENV=$TANDEM_ENV" \
    CALIBRATION_VERIFIED=1 DEFER_TRAINING_CHECK=1)
printf 'train %s\n' "$TRAIN" > "$RUN_ROOT/training-submitted.txt"
VERIFY=$(sbatch "${CPU[@]}" --array=0-1 --dependency="aftercorr:$TRAIN" \
    --job-name=pilot-train-check --output="$REPO/logs/pilot-train-check-%A_%a.out" \
    slurm/shorthand-check.sbatch "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" PHASE=training)
BASE=$(sbatch "${GPU[@]}" --exclude="$EVAL_EXCLUDE" --cpus-per-task=4 --mem=32G \
    --time=06:00:00 --dependency="afterok:$GATE" --job-name=pilot-base \
    --output="$REPO/logs/pilot-base-%j.out" slurm/shorthand-eval.sbatch \
    "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" PHASE=base)
EVAL=$(sbatch "${GPU[@]}" --array=0-1 --exclude="$EVAL_EXCLUDE" --cpus-per-task=4 --mem=32G \
    --time=06:00:00 --dependency="aftercorr:$VERIFY" --job-name=pilot-eval \
    --output="$REPO/logs/pilot-eval-%A_%a.out" slurm/shorthand-eval.sbatch \
    "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" PHASE=trained)
REVIEW=$(sbatch "${CPU[@]}" --mem=2G --time=00:15:00 --dependency="afterok:$BASE:$EVAL" \
    --job-name=pilot-review --output="$REPO/logs/pilot-review-%j.out" \
    slurm/shorthand-review.sbatch "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" EXPECTED_COMPARISONS=8)
printf 'setup %s\ncalibration %s\ngate %s\ntrain %s\nverify %s\nbase %s\neval %s\nreview %s\n' \
    "$SETUP_JOB" "$CALIBRATION_JOB" "$GATE" "$TRAIN" "$VERIFY" "$BASE" "$EVAL" "$REVIEW" | tee "$RUN_ROOT/jobs.txt"
