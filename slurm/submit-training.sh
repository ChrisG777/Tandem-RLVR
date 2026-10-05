#!/usr/bin/env bash
# CSAIL: submit both full arms alongside smoke jobs, guarded by CPU allocations.
set -euo pipefail
cd "${REPO:?}"
SITE=("REPO=$REPO" "TANDEM_ENV=$TANDEM_ENV" "RUN_ROOT=$RUN_ROOT"
      "DATA_ROOT=$DATA_ROOT" "HF_HOME=$HF_HOME" "UV_CACHE_DIR=$UV_CACHE_DIR")
GPU=(--partition=vision-shared-h200,vision-shared-h100 --account=vision-torralba
     --qos=shared-if-available --export=NIL --chdir="$REPO"
     --dependency="afterok:${SETUP_JOB:?}" --kill-on-invalid-dep=yes)
mkdir -p "$RUN_ROOT"
for ARM in grpo tandem; do
    SMOKE=$(sbatch --parsable "${GPU[@]}" --job-name="$ARM-fresh-smoke" --time=02:00:00 \
        --output="$REPO/logs/$ARM-smoke-%j.out" slurm/train-figure2.sbatch "${SITE[@]}" ARM="$ARM" MODE=smoke)
    FULL=$(sbatch --parsable --hold "${GPU[@]}" --job-name="$ARM-fresh-full" \
        --output="$REPO/logs/$ARM-full-%j.out" slurm/train-figure2.sbatch "${SITE[@]}" ARM="$ARM" MODE=full)
    WATCH=$(sbatch --parsable --job-name="$ARM-smoke-watch" --partition=tig-cpu --account=csail \
        --qos=tig-main --cpus-per-task=1 --mem=1G --time=24:00:00 --signal=B:TERM@60 \
        --export=NIL --chdir="$REPO" --output="$REPO/logs/$ARM-watch-%j.out" \
        slurm/watch-training.sh "REPO=$REPO" "UV_CACHE_DIR=$UV_CACHE_DIR" \
        "WATCH_ENV=$WATCH_ENV" "RUN_ROOT=$RUN_ROOT" "ARM=$ARM" "SMOKE=$SMOKE" "FULL=$FULL")
    # Only environment availability and monitor startup gate full training.
    # There is deliberately no dependency on smoke success.
    scontrol update JobId="$FULL" Dependency="afterok:$SETUP_JOB,after:$WATCH"
    scontrol release "$FULL"
    echo "$ARM $SMOKE $FULL $WATCH" | tee -a "$RUN_ROOT/jobs.txt"
done
