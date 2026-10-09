#!/usr/bin/env bash
# One CSAIL task campaign, conditional on an already submitted calibration.
# Select/validate live routes before invoking; this script does not select a site.
set -euo pipefail
cd "${REPO:?}"
: "${RUN_ROOT:?}" "${DATA_ROOT:?}" "${TANDEM_ENV:?}" "${EVAL_ENV:?}"
: "${CALIBRATION_JOB:?}" "${PILOT_TASKS:?}" "${PILOT_BUDGETS:?}"
: "${TRAIN_PARTITIONS:?}" "${EVAL_PARTITIONS:?}"
[[ "$PILOT_TASKS" = ruletaker_shared || "$PILOT_TASKS" = rearc_objects ]]
mkdir "$RUN_ROOT/submission-lock" # Prevent duplicate continuations, even after a partial submission.
SITE=("REPO=$REPO" "RUN_ROOT=$RUN_ROOT" "DATA_ROOT=$DATA_ROOT"
      "PILOT_TASKS=$PILOT_TASKS" "PILOT_BUDGETS=$PILOT_BUDGETS")
COMMON=(--parsable --export=NIL --chdir="$REPO" --nice=1000 --kill-on-invalid-dep=yes)
GPU=("${COMMON[@]}" --account=vision-torralba --qos=shared-if-available --gpus=1 --requeue)
CPU=("${COMMON[@]}" --partition=tig-cpu --account=csail --qos=tig-main
     --cpus-per-task=1 --mem=4G --time=01:00:00)
record() { printf '%s %s\n' "$1" "$2" | tee -a "$RUN_ROOT/jobs.txt"; }
record calibration "$CALIBRATION_JOB"
TRAIN=$(sbatch "${GPU[@]}" --array=0-1 --partition="$TRAIN_PARTITIONS" \
    '--constraint=nvidia_h200|nvidia_h100_80gb_hbm3|nvidia_h100_nvl|nvidia_a100-sxm4-80gb' \
    --cpus-per-task=6 --mem=144G --time=06:00:00 --dependency="afterok:$CALIBRATION_JOB" \
    --job-name="$PILOT_TASKS-solo" --output="$REPO/logs/concept-train-%A_%a.out" \
    slurm/shorthand-train.sbatch "${SITE[@]}" "TANDEM_ENV=$TANDEM_ENV" \
    CALIBRATION_VERIFIED=1 DEFER_TRAINING_CHECK=1)
record train "$TRAIN"
VERIFY=$(sbatch "${CPU[@]}" --array=0-1 --dependency="aftercorr:$TRAIN" \
    --job-name=concept-train-check --output="$REPO/logs/concept-check-%A_%a.out" \
    slurm/shorthand-check.sbatch "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" \
    UV_BIN_DIR=/data/scratch/chrisge/.local/bin PHASE=training)
record verify "$VERIFY"
BASE=$(sbatch "${GPU[@]}" --partition="$EVAL_PARTITIONS" --time=06:00:00 \
    --dependency="afterok:$CALIBRATION_JOB" --job-name="$PILOT_TASKS-base" \
    --output="$REPO/logs/concept-base-%j.out" slurm/shorthand-eval.sbatch \
    "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" PHASE=base)
record base "$BASE"
EVAL=$(sbatch "${GPU[@]}" --array=0-1 --partition="$EVAL_PARTITIONS" --time=06:00:00 \
    --dependency="aftercorr:$VERIFY" --job-name="$PILOT_TASKS-eval" \
    --output="$REPO/logs/concept-eval-%A_%a.out" slurm/shorthand-eval.sbatch \
    "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" PHASE=trained)
record eval "$EVAL"
REVIEW=$(sbatch "${CPU[@]}" --mem=2G --time=00:15:00 --dependency="afterok:$BASE:$EVAL" \
    --job-name=concept-review --output="$REPO/logs/concept-review-%j.out" \
    slurm/shorthand-review.sbatch "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" EXPECTED_COMPARISONS=8)
record review "$REVIEW"
