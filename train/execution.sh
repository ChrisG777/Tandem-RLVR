# Source from an experiment wrapper; consumes no scheduler resources itself.
# configure_training_execution MODEL_FAMILY MODE
# MODEL_FAMILY: qwen3 | qwen35. MODE: baseline | optimized.
# Sets EXECUTION_EXTRA (Hydra argv array) and execution-only environment settings.
# Does not change datasets, rewards, model weights, RL batches, decoding sampling,
# response budgets, checkpoint cadence, or the number of optimizer steps.
configure_training_execution() {
    local family=${1:?model family required} mode=${2:?execution mode required}
    EXECUTION_EXTRA=()
    case "$family" in qwen3|qwen35) ;; *) echo "Unsupported model family: $family" >&2; return 2;; esac
    case "$mode" in baseline) return 0;; optimized) ;; *) echo "Unsupported execution mode: $mode" >&2; return 2;; esac
    if [ "$family" = qwen35 ]; then
        test -x "${CUDA_HOME:?Qwen3.5 needs the qualified CUDA toolkit}/bin/nvcc" || return 2
    fi
    export ENFORCE_EAGER=False OMP_NUM_THREADS=${TRAIN_OMP_THREADS:-4}
    export VERL_USE_FLA_CROSS_ENTROPY=1
    EXECUTION_EXTRA+=(
        +actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=True
        actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
        actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True
        actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=512
        'actor_rollout_ref.actor.optim.override_optimizer_config={fused:True}'
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config={mode:0,cudagraph_mode:FULL_DECODE_ONLY}')
    if [ -n "${TRAIN_PROFILE_DIR:-}" ]; then
        export QWEN35_PROFILE_DIR=$TRAIN_PROFILE_DIR
        EXECUTION_EXTRA+=(actor_rollout_ref.model.external_lib=train.profile_qwen35)
    fi
}
