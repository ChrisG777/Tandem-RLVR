#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
export PILOT_ENTRYPOINT=drift-eval.sbatch
export CKPT_DIR="${RUN_ROOT:?}/${MODEL_KEY:?}-lr${LR:-1e-5}-s${SEED:-42}"
source "$REPO/slurm/shorthand-runtime.sh"
[ "${SLURM_RESTART_COUNT:-0}" -le 12 ] || exit 1
export PATH="${UV_BIN_DIR:?}:$PATH" HF_HUB_OFFLINE=1
export TANDEM_ENV_BIN="${TANDEM_ENV:?}/bin"
export XDG_CACHE_HOME="/tmp/drift-eval-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
source slurm/training-env.sh
run_python() { uv run --no-project --offline --python "$TANDEM_ENV_BIN/python" python "$@"; }
run_python eval/drift.py --select-run "$CKPT_DIR"
for phase in base selected; do
    if [ "$phase" = base ]; then manifest="$DATA_ROOT/$MODEL_KEY.json"; else manifest="$CKPT_DIR/selected.json"; fi
    MODEL=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$manifest")
    run_python eval/drift.py --model "$MODEL" --data "$DATA_ROOT/test.parquet" --out "$CKPT_DIR/eval/$phase.json" &
    wait $!
done
