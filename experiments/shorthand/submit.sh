#!/usr/bin/env bash
# Submit one calibrated pilot campaign. Routes/resources are explicit site inputs.
# *_SBATCH_ARGS are whitespace-separated flags (no shell evaluation or embedded spaces).
set -euo pipefail
cd "${REPO:?}"
: "${RUN_ROOT:?}" "${DATA_ROOT:?}" "${TANDEM_ENV:?}" "${EVAL_ENV:?}"
: "${CALIBRATION_JOB:?}" "${PILOT_TASKS:?}" "${PILOT_BUDGETS:?}"
: "${CPU_SBATCH_ARGS:?}" "${TRAIN_SBATCH_ARGS:?}" "${EVAL_SBATCH_ARGS:?}" "${UV_BIN_DIR:?}"
read -ra TASKS <<< "$PILOT_TASKS"
read -ra BUDGETS <<< "$PILOT_BUDGETS"
[ "${#TASKS[@]}" -gt 0 ] || { echo 'Require at least one task' >&2; exit 2; }
[ "${#BUDGETS[@]}" = 2 ] || { echo 'Require exactly two response budgets' >&2; exit 2; }
[[ "${BUDGETS[0]}" =~ ^[1-9][0-9]*$ && "${BUDGETS[1]}" =~ ^[1-9][0-9]*$ ]] || exit 2
[ "${BUDGETS[0]}" -lt "${BUDGETS[1]}" ] || exit 2
for task in "${TASKS[@]}"; do
    case "$task" in manipulate_matrix|string_manipulation|ruletaker_shared|rearc_objects) ;;
        *) echo "Unknown pilot task: $task" >&2; exit 2;; esac
done
# Current pilot verification/evaluation protocol selects step 100.
[ "${TOTAL_STEPS:-100}" = 100 ] || { echo 'This campaign evaluates step 100' >&2; exit 2; }
read -ra CPU <<< "$CPU_SBATCH_ARGS"
read -ra TRAIN_ARGS <<< "$TRAIN_SBATCH_ARGS"
read -ra EVAL_ARGS <<< "$EVAL_SBATCH_ARGS"
COMMON=(--parsable --export=NIL --chdir="$REPO" --kill-on-invalid-dep=yes)
SITE=("REPO=$REPO" "RUN_ROOT=$RUN_ROOT" "DATA_ROOT=$DATA_ROOT"
      "UV_BIN_DIR=$UV_BIN_DIR" "PILOT_TASKS=$PILOT_TASKS" "PILOT_BUDGETS=$PILOT_BUDGETS")
for name in HF_HOME UV_CACHE_DIR APPTAINER_IMAGE APPTAINER_BIN_DIR CONTAINER_BIND \
            PILOT_CONTEXT PILOT_MAX_PROMPT PILOT_PPO_MAX_TOKEN PILOT_LENGTH_PENALTY MAX_RESTARTS; do
    if [ -n "${!name:-}" ]; then SITE+=("$name=${!name}"); fi
done
mkdir -p "$RUN_ROOT" "$REPO/logs"
[ ! -e "$RUN_ROOT/training-submitted.txt" ] || { echo 'Campaign already submitted'; exit 1; }
mkdir "$RUN_ROOT/submission-lock" # Retain on partial submission; inspect recorded jobs before retrying.
record() { printf '%s %s\n' "$1" "$2" | tee -a "$RUN_ROOT/jobs.txt"; }
record calibration "$CALIBRATION_JOB"
DEPENDENCY="afterok:$CALIBRATION_JOB"
if [ -n "${SETUP_JOB:-}" ]; then DEPENDENCY="$DEPENDENCY:$SETUP_JOB"; fi
PREFIX=experiments/shorthand/slurm
GATE=$(sbatch "${COMMON[@]}" "${CPU[@]}" --dependency="$DEPENDENCY" \
    --job-name=pilot-calibration-check --output="$REPO/logs/pilot-calibration-check-%j.out" \
    "$PREFIX/shorthand-check.sbatch" "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV")
record gate "$GATE"
LAST=$((${#TASKS[@]} * 2 - 1))
TRAIN=$(sbatch "${COMMON[@]}" "${TRAIN_ARGS[@]}" --requeue --array="0-$LAST" \
    --dependency="afterok:$GATE" --job-name=pilot-solo --output="$REPO/logs/pilot-solo-%A_%a.out" \
    "$PREFIX/shorthand-train.sbatch" "${SITE[@]}" "TANDEM_ENV=$TANDEM_ENV" \
    CALIBRATION_VERIFIED=1 DEFER_TRAINING_CHECK=1)
record train "$TRAIN"
printf 'train %s\n' "$TRAIN" > "$RUN_ROOT/training-submitted.txt"
VERIFY=$(sbatch "${COMMON[@]}" "${CPU[@]}" --array="0-$LAST" --dependency="aftercorr:$TRAIN" \
    --job-name=pilot-training-check --output="$REPO/logs/pilot-training-check-%A_%a.out" \
    "$PREFIX/shorthand-check.sbatch" "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" PHASE=training)
record verify "$VERIFY"
BASE=$(sbatch "${COMMON[@]}" "${EVAL_ARGS[@]}" --requeue --dependency="afterok:$GATE" \
    --job-name=pilot-base --output="$REPO/logs/pilot-base-%j.out" \
    "$PREFIX/shorthand-eval.sbatch" "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" PHASE=base)
record base "$BASE"
EVAL=$(sbatch "${COMMON[@]}" "${EVAL_ARGS[@]}" --requeue --array="0-$LAST" \
    --dependency="aftercorr:$VERIFY" --job-name=pilot-eval --output="$REPO/logs/pilot-eval-%A_%a.out" \
    "$PREFIX/shorthand-eval.sbatch" "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" PHASE=trained)
record eval "$EVAL"
REVIEW=$(sbatch "${COMMON[@]}" "${CPU[@]}" --dependency="afterok:$BASE:$EVAL" \
    --job-name=pilot-review --output="$REPO/logs/pilot-review-%j.out" \
    "$PREFIX/shorthand-review.sbatch" "${SITE[@]}" "TANDEM_ENV=$EVAL_ENV" \
    "EXPECTED_COMPARISONS=$((${#TASKS[@]} * 8))")
record review "$REVIEW"
