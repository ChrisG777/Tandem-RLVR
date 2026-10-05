# Batch jobs supply all paths explicitly. This file is sourced by existing launchers.
export PATH="${TANDEM_ENV_BIN:?}:$PATH"
# Ray 2.55's automatic uv hook rejects verl's runtime_env.working_dir=None.
# Workers already share this explicit uv-managed interpreter on the same node.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# One trainable policy GPU; Tandem additionally reserves its frozen junior GPU.
export TRAIN_GPUS=${TRAIN_GPUS:-1}
# Single-node rollout storage: one storage CPU + one controller CPU, plus
# three CPUs in verl's policy placement group and one for scheduling progress.
export TQ_STORAGE_UNITS=${TQ_STORAGE_UNITS:-1}
export DATA_WORKERS=${DATA_WORKERS:-2}
export AGENT_WORKERS=${AGENT_WORKERS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Avoid Ray reserving 30% of a large host's memory for small rollout tensors.
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4294967296
