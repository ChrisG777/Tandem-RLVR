# Source after setting JOB_ENTRYPOINT (repo-relative) and optionally one of:
# RESUME_CHECKPOINT_ROOT, RESUME_PROGRESS_FILE, RESUME_PROGRESS_DIR.
# Host handles USR1/requeue; applications restore their own durable progress.
export PATH="/usr/bin:/bin:${PATH:-}"
resume_before_walltime() {
    [ "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_RESTARTS:-12}" ] || return 1
    if [ -n "${RESUME_CHECKPOINT_ROOT:-}" ]; then
        local step
        step=$(cat "$RESUME_CHECKPOINT_ROOT/latest_checkpointed_iteration.txt") || return 1
        [[ "$step" =~ ^[1-9][0-9]*$ ]] || return 1
        test -s "$RESUME_CHECKPOINT_ROOT/global_step_$step/data.pt" || return 1
        local kind
        for kind in model optim extra_state; do
            test -s "$RESUME_CHECKPOINT_ROOT/global_step_$step/actor/${kind}_world_size_1_rank_0.pt" || return 1
        done
    elif [ -n "${RESUME_PROGRESS_FILE:-}" ]; then
        test -s "$RESUME_PROGRESS_FILE" || return 1
    elif [ -n "${RESUME_PROGRESS_DIR:-}" ]; then
        [ -d "$RESUME_PROGRESS_DIR" ] || return 1
        [ -n "$(find "$RESUME_PROGRESS_DIR" -maxdepth 1 -name '*.json*' -type f -size +0c -print -quit)" ] || return 1
    else
        return 1
    fi
    local job="${SLURM_JOB_ID:?}"
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
        job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID:?}"
    fi
    echo "Walltime continuation: requeue $job from saved progress"
    scontrol requeue "$job"
}
if [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ] && [ -n "${RESUME_CHECKPOINT_ROOT:-}${RESUME_PROGRESS_FILE:-}${RESUME_PROGRESS_DIR:-}" ]; then
    trap resume_before_walltime USR1
fi
if [ -n "${APPTAINER_IMAGE:-}" ] && [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ]; then
    export CODE_COMMIT=$(git rev-parse HEAD)
    export TANDEM_CONTAINER_ACTIVE=1
    GPU_FLAG=
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then GPU_FLAG=--nv; fi
    "${APPTAINER_BIN_DIR:-/orcd/software/core/001/pkg/apptainer/1.4.2/bin}/apptainer" exec \
        ${GPU_FLAG} --bind "${CONTAINER_BIND:?}" "$APPTAINER_IMAGE" \
        bash "$REPO/${JOB_ENTRYPOINT:?}" &
    CHILD=$!
    wait "$CHILD"
    exit $?
fi
if [ "${TANDEM_CONTAINER_ACTIVE:-0}" = 1 ]; then
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
else
    export CODE_COMMIT=$(git rev-parse HEAD)
fi
