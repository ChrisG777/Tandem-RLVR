#!/usr/bin/env bash
# All resource/account/partition options are explicit submission arguments.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
cd "${REPO:?}"
export PATH="${UV_BIN_DIR:?}:$PATH"
export APPS_GRADER_RUNTIME="${RUN_ROOT:?}/grader"
export APPS_BWRAP="$APPS_GRADER_RUNTIME/bwrap-libs/ld-linux-x86-64.so.2 --library-path $APPS_GRADER_RUNTIME/bwrap-libs $APPS_GRADER_RUNTIME/bwrap"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="/tmp/apps-cache-${SLURM_JOB_ID:-$$}"
export XDG_CONFIG_HOME="$XDG_CACHE_HOME/config" VLLM_CONFIG_ROOT="$XDG_CACHE_HOME/config/vllm"
mkdir -p "$VLLM_CONFIG_ROOT"
PHASE=${1:-${CODING_PHASE:?}}
if [ "$PHASE" = setup ]; then
    export UV_PROJECT_ENVIRONMENT="$RUN_ROOT/tools"
    uv sync --project experiments/coding/env --locked
    uv run --project experiments/coding/env --locked python -m experiments.coding.setup_grader --runtime "$APPS_GRADER_RUNTIME"
    uv run --project experiments/coding/env --locked python -m experiments.coding.campaign preflight
    if [ ! -f "$DATA_ROOT/manifest.json" ]; then
        uv run --project experiments/coding/env --locked python -m experiments.coding.build_apps --out-dir "$DATA_ROOT" --revision "${DATA_REVISION:?}"
    fi
    uv run --project experiments/coding/env --locked python -m experiments.coding.campaign panel --data "$DATA_ROOT" --out "$RUN_ROOT/panel.json"
    uv run --project experiments/coding/env --locked python -m experiments.coding.campaign protocol --data "$DATA_ROOT" --base "$RUN_ROOT/base.json" --out "$RUN_ROOT/protocol.json"
    exit
fi
export JOB_ENTRYPOINT=experiments/coding/job.sbatch
if [ "$PHASE" = train ]; then
    export CKPT_DIR="$RUN_ROOT/${ARM:?}-${MODE:?}"
    export RESUME_CHECKPOINT_ROOT="$CKPT_DIR"
else
    export RESUME_PROGRESS_DIR="$RUN_ROOT/calibration"
    if [ "$PHASE" = evaluate ]; then export RESUME_PROGRESS_DIR="$RUN_ROOT/results/${ARM:?}"; fi
fi
# runtime.sh relaunches this file inside the container, so preserve phase in env.
export CODING_PHASE="$PHASE"
source "$REPO/slurm/runtime.sh"
run_python() { uv run --no-project --offline --python "${TANDEM_ENV:?}/bin/python" python "$@"; }
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
BASE_MODEL=$(run_python -c 'import json,os; print(json.load(open(os.environ["RUN_ROOT"]+"/base.json"))["base"]["path"])')
case "$PHASE" in
    calibration)
        mkdir -p "$RUN_ROOT/calibration"
        CALIBRATION_EXTRA=()
        if [ -n "${NATIVE_CONCURRENCY:-}" ]; then CALIBRATION_EXTRA+=(--native-concurrency "$NATIVE_CONCURRENCY"); fi
        run_python -m experiments.coding.campaign evaluate --phase "${EVAL_PHASE:?}" --model "$BASE_MODEL" --junior "$BASE_MODEL" --data "$DATA_ROOT" --split validation --out "$RUN_ROOT/calibration/$EVAL_PHASE.json" --decoding training --n 8 "${CALIBRATION_EXTRA[@]}" &
        wait $!
        ;;
    gate)
        run_python -m experiments.coding.campaign calibration --data "$DATA_ROOT" --solo "$RUN_ROOT/calibration/solo.json" --word "$RUN_ROOT/calibration/word-handoff.json" --out "$RUN_ROOT/calibration/report.json"
        ;;
    train)
        ADMISSION=()
        if [ "${ALLOW_PENDING_CALIBRATION:-0}" = 1 ]; then ADMISSION+=(--allow-pending); fi
        # Profiling is a bounded, disposable experiment, never production admission.
        if [ "$MODE" != profile ]; then
            run_python -m experiments.coding.campaign admit-training --report "$RUN_ROOT/calibration/report.json" "${ADMISSION[@]}"
        fi
        run_python -m experiments.coding.campaign preflight
        export TANDEM_ENV_BIN="$TANDEM_ENV/bin" TANDEM_ENV_FILE="$REPO/slurm/training-env.sh"
        export BASE_MODEL POLICY_MODEL="$BASE_MODEL" JUNIOR_MODEL="$BASE_MODEL"
        export CKPT_ROOT="$RUN_ROOT" PROJECT_NAME=tandem-apps EXP_NAME="$ARM-$MODE"
        export TRAIN_FILES="$DATA_ROOT/train.parquet" MAX_PROMPT=4096 MAX_RESPONSE=6000
        export TOTAL_STEPS=200 TOTAL_EPOCHS=2 TEST_FREQ=20 SAVE_FREQ=${SAVE_FREQ:-20} PERSIST_HF_INTERVAL=20
        export TRAIN_BATCH=16 MINI_BATCH=8 ROLLOUT_N=8 LR=1e-6 CLIP_RATIO=0.2 TRAIN_GPUS=1
        export ROLLOUT_TEMP=0.6 VAL_N=4 VAL_TEMP=0.7 VAL_TOP_P=0.8 VAL_TOP_K=20 VAL_DO_SAMPLE=True
        export PROB_PRIMARY=0.5 SELECTION=word MAX_GAP_TOKENS=32 JR_TKN_WEIGHT=0
        export FROZEN_GPU=0
        export PPO_MAX_TOKEN=12288 ROLLOUT_GPU_UTIL=0.65 TQ_STORAGE_UNITS=1 DATA_WORKERS=2 AGENT_WORKERS=4
        export WANDB_MODE=offline HYDRA_FULL_ERROR=1
        export RAY_TMPDIR="/tmp/apps-ray-${SLURM_JOB_ID}" XDG_CACHE_HOME="/tmp/apps-cache-${SLURM_JOB_ID}"
        export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
        mkdir -p "$CKPT_DIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME"
        export VERL_FILE_LOGGER_PATH="$CKPT_DIR/metrics-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.jsonl"
        EXTRA=()
        if [ "$MODE" = smoke ]; then
            export TOTAL_STEPS=3 TEST_FREQ=3 SAVE_FREQ=3 PERSIST_HF_INTERVAL=3
            EXTRA+=("data.val_max_samples=${SMOKE_VAL_MAX_SAMPLES:-8}")
        elif [ "$MODE" = profile ]; then
            export TOTAL_STEPS=${PROFILE_STEPS:-3} TEST_FREQ=${PROFILE_TEST_FREQ:--1} SAVE_FREQ=-1 PERSIST_HF_INTERVAL=20
            [[ "$TOTAL_STEPS" =~ ^[1-9][0-9]*$ ]] && [ "$TOTAL_STEPS" -le 5 ] || exit 2
        elif [ "$MODE" != full ]; then exit 2; fi
        cp "$RUN_ROOT/base.json" "$CKPT_DIR/initialization.json"
        cp "$DATA_ROOT/manifest.json" "$CKPT_DIR/data-manifest.json"
        printf '%s\n' "$CODE_COMMIT" > "$CKPT_DIR/code-${SLURM_JOB_ID}.txt"
        LAUNCHER=train/vanilla_grpo.sh
        if [ "$ARM" = tandem ]; then LAUNCHER=train/tandem_grpo.sh; fi
        bash "$LAUNCHER" "data.val_files=[$DATA_ROOT/validation.parquet]" data.seed=42 \
            "reward.custom_reward_function.path=$REPO/experiments/coding/reward.py" \
            'trainer.logger=[console,file]' "ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK" \
            actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.top_p=1 actor_rollout_ref.rollout.top_k=-1 \
            actor_rollout_ref.model.lora_rank=0 \
            actor_rollout_ref.rollout.max_model_len=12288 trainer.resume_mode=auto \
            "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_SEQS:-32}" \
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=${ROLLOUT_KV_BYTES:-51539607552}" \
            "hydra.run.dir=$CKPT_DIR/hydra-${SLURM_JOB_ID}" "${EXTRA[@]}" &
        wait $!
        if [ "$MODE" = profile ]; then exit 0; fi
        run_python train/figure2_checkpoint.py --root "$CKPT_DIR" --arm "$ARM" --steps "$TOTAL_STEPS" --mode "$MODE" --validation-metric val-core/apps/acc/best@4/mean
        ;;
    select)
        run_python -m experiments.coding.campaign select --root "$RUN_ROOT"
        ;;
    evaluate)
        MODEL=$(run_python -c 'import json,os; print(json.load(open(os.environ["RUN_ROOT"]+"/models.json"))[os.environ["ARM"]]["path"])')
        N=8
        if [ "$EVAL_PHASE" = solo ]; then N=32; fi
        run_python -m experiments.coding.campaign evaluate --phase "$EVAL_PHASE" --model "$MODEL" --junior "$BASE_MODEL" --data "$DATA_ROOT" --out "$RUN_ROOT/results/$ARM/$EVAL_PHASE.json" --n "$N" --solo "$RUN_ROOT/results/$ARM/solo.json" &
        wait $!
        ;;
    plot)
        run_python eval/figure2.py --results "$RUN_ROOT/results" --out "$RUN_ROOT/results/apps-figure2" --panel-manifest "$RUN_ROOT/panel.json"
        ;;
    *) exit 2;;
esac
