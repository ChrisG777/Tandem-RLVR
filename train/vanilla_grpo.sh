#!/usr/bin/env bash
set -xeuo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
ENV_FILE=${TANDEM_ENV_FILE:-$HERE/env.sh}
[ -f "$ENV_FILE" ] || { echo "[error] $ENV_FILE not found" >&2; exit 1; }
source "$ENV_FILE"

# Keep this arm solo even if the calling shell previously enabled tandem.
unset VLLM_TANDEM_CONFIG VLLM_TANDEM_ALL_GPUS

POLICY_MODEL=${POLICY_MODEL:-$BASE_MODEL}
GPUS=${GPUS:-0,1}
TRAIN_GPUS=${TRAIN_GPUS:-2}

LR=${LR:-1e-6}
CLIP_RATIO=${CLIP_RATIO:-0.2}
TRAIN_BATCH=${TRAIN_BATCH:-16}
MINI_BATCH=${MINI_BATCH:-8}
MAX_PROMPT=${MAX_PROMPT:-1024}
MAX_RESPONSE=${MAX_RESPONSE:-3000}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-0.6}
PPO_MAX_TOKEN=${PPO_MAX_TOKEN:-10000}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TOTAL_STEPS=${TOTAL_STEPS:-null}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMP=${VAL_TEMP:-0.7}
VAL_TOP_P=${VAL_TOP_P:-0.8}
VAL_TOP_K=${VAL_TOP_K:-20}
VAL_N=${VAL_N:-4}
ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL:-0.8}
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
OFFLOAD=${OFFLOAD:-True}
EXP_NAME=${EXP_NAME:-vanilla_grpo_qwen3_4b_deepscaler}
CKPT_DIR=${CKPT_DIR:-$CKPT_ROOT/${EXP_NAME}}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-1}

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
if [ -z "${SLURM_JOB_ID:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPUS; fi

train_files=${TRAIN_FILES:-$DATA_ROOT/deepscaler/train.parquet}
val_files="[$DATA_ROOT/deepscaler/heldout.parquet]"
reward_fn=$REPO/reward/math_boxed_reward.py
max_num_tokens=$(( MAX_PROMPT + MAX_RESPONSE + 1 ))

uv run --python "${TANDEM_ENV_BIN:?}/python" --no-project --offline python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${TRAIN_BATCH} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESPONSE} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$POLICY_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMP} \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMP} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    model_engine=dp \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="$reward_fn" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='[console,wandb]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${TRAIN_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP} \
    +trainer.persist_hf_model=True \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    "$@"
