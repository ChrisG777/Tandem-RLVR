# Batch jobs supply all paths explicitly. This file is sourced by existing launchers.
export PATH="${TANDEM_ENV_BIN:?}:$PATH"
# Ray 2.55's automatic uv hook rejects verl's runtime_env.working_dir=None.
# Workers already share this explicit uv-managed interpreter on the same node.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
