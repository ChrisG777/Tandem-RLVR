#!/usr/bin/env bash
set -euo pipefail
export ROLLOUT_N=16
export EXP_NAME=${EXP_NAME:-vanilla_grpo_qwen3_4b_deepscaler_g16}
exec bash "$(cd "$(dirname "$0")" && pwd)/vanilla_grpo.sh" "$@"
