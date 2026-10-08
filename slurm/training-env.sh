# Batch jobs supply all paths explicitly. This file is sourced by existing launchers.
export PATH="${TANDEM_ENV_BIN:?}:$PATH"
# AFS credentials can expire during queued/requeued CSAIL jobs. Runtime config
# belongs with the job-local caches, not in the authenticated home directory.
export XDG_CONFIG_HOME="${XDG_CACHE_HOME:-/tmp/tandem-${SLURM_JOB_ID:-$$}}/config"
export VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
mkdir -p "$VLLM_CONFIG_ROOT"
# Ray 2.55's automatic uv hook rejects verl's runtime_env.working_dir=None.
# Workers already share this explicit uv-managed interpreter on the same node.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# One trainable policy GPU; Tandem additionally reserves its frozen junior GPU.
export TRAIN_GPUS=${TRAIN_GPUS:-1}
# Leave room for the colocated FSDP policy before vLLM reserves its cache.
# The vanilla launcher's 0.8 fails startup on an 80 GB GPU with one training rank.
export ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL:-0.65}
# Single-node rollout storage: one storage CPU + one controller CPU, plus
# three CPUs in verl's policy placement group and one for scheduling progress.
export TQ_STORAGE_UNITS=${TQ_STORAGE_UNITS:-1}
export DATA_WORKERS=${DATA_WORKERS:-2}
export AGENT_WORKERS=${AGENT_WORKERS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Avoid Ray reserving 30% of a large host's memory for small rollout tensors.
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4294967296
