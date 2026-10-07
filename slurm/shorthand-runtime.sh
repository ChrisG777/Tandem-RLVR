# Sourced after site arguments are exported. Containers share all persistent paths.
# The host shell handles Slurm's pre-walltime signal; the application resumes itself.
export PATH="/usr/bin:/bin:${PATH:-}"
if [ -n "${APPTAINER_IMAGE:-}" ] && [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ]; then
    export CODE_COMMIT=$(git rev-parse HEAD)
    export TANDEM_CONTAINER_ACTIVE=1
    GPU_FLAG=
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then GPU_FLAG=--nv; fi
    resume_before_walltime() {
        [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ] || return 1
        read -ra TASKS <<< "${PILOT_TASKS:-manipulate_matrix string_manipulation}"
        read -ra BUDGETS <<< "${PILOT_BUDGETS:-256 1024}"
        ROOT="$RUN_ROOT/train/${TASKS[$((SLURM_ARRAY_TASK_ID / 2))]}-b${BUDGETS[$((SLURM_ARRAY_TASK_ID % 2))]}"
        STEP=$(cat "$ROOT/latest_checkpointed_iteration.txt") || return 1
        [[ "$STEP" =~ ^[1-9][0-9]*$ ]] || return 1
        test -s "$ROOT/global_step_$STEP/data.pt" || return 1
        for KIND in model optim extra_state; do
            test -s "$ROOT/global_step_$STEP/actor/${KIND}_world_size_1_rank_0.pt" || return 1
        done
        echo "Walltime continuation: requeue $SLURM_JOB_ID from checkpoint $STEP"
        scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    }
    if [ "$PILOT_ENTRYPOINT" = shorthand-train.sbatch ]; then
        trap resume_before_walltime USR1
    fi
    "${APPTAINER_BIN_DIR:-/orcd/software/core/001/pkg/apptainer/1.4.2/bin}/apptainer" exec \
        ${GPU_FLAG} --bind "${CONTAINER_BIND:?}" "$APPTAINER_IMAGE" \
        bash "$REPO/slurm/${PILOT_ENTRYPOINT:?}" &
    CHILD=$!
    wait "$CHILD"
    exit $?
fi
if [ "${TANDEM_CONTAINER_ACTIVE:-0}" = 1 ]; then
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
else
    export CODE_COMMIT=$(git rev-parse HEAD)
fi
