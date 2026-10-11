#!/usr/bin/env bash
# Select account/partition/QoS when submitting; both engines share one GPU.
#SBATCH --job-name=tandem-figure2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/figure2-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
# Explicit KEY=VALUE arguments also support sbatch --export=NIL on CSAIL.
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?set REPO to the git checkout}"
export JOB_ENTRYPOINT=slurm/figure2.sbatch
export RESULTS_ROOT=${RESULTS_ROOT:-results/figure2}
case "${MODE:-full}" in
    base) export RESUME_PROGRESS_FILE="$RESULTS_ROOT/base/solo.json.progress.json";;
    *-solo) export RESUME_PROGRESS_FILE="$RESULTS_ROOT/${MODE%-solo}/solo.json.progress.json";;
    *-handoff) export RESUME_PROGRESS_FILE="$RESULTS_ROOT/${MODE%-handoff}/handoff.json.progress.json";;
esac
source "$REPO/slurm/runtime.sh"
[ "${SLURM_RESTART_COUNT:-0}" -le "${MAX_RESTARTS:-12}" ] || { echo 'Preemption retry limit reached'; exit 1; }
export PATH="${UV_BIN_DIR:-${HOME:-/tmp}/.local/bin}:/data/scratch/${SLURM_JOB_USER:-${USER:-}}/.local/bin:${PATH:-/usr/bin:/bin}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tandem-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/tandem-cache-${SLURM_JOB_ID}"
export XDG_CONFIG_HOME="$XDG_CACHE_HOME/config" VLLM_CONFIG_ROOT="$XDG_CACHE_HOME/config/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer-workspace"
mkdir -p "$XDG_CACHE_HOME"
unset VLLM_TANDEM_CONFIG VLLM_TANDEM_ALL_GPUS
TANDEM_ENV=${TANDEM_ENV:-$REPO/.venv}
run_python() { uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python "$@" & wait "$!"; }
ROOT=${RESULTS_ROOT:-results/figure2}
mkdir -p "$ROOT"
if [ -n "${CODE_COMMIT:-}" ]; then
    echo "$CODE_COMMIT" > "$ROOT/code-${SLURM_JOB_ID}.txt"
else
    git rev-parse HEAD > "$ROOT/code-${SLURM_JOB_ID}.txt"
fi
cp "$ROOT/code-${SLURM_JOB_ID}.txt" "$ROOT/code-commit.txt"
cp "${MODELS_JSON:?}" "$ROOT/models-${SLURM_JOB_ID}.json"
nvidia-smi > "$ROOT/gpus-${SLURM_JOB_ID}.txt"
MODE=${MODE:-full}
case "$MODE" in base|full|base-solo|junior-solo|grpo-solo|tandem-solo|base-handoff|grpo-handoff|tandem-handoff) ;; *) echo 'Unknown evaluation mode' >&2; exit 2;; esac
MODEL_PATHS="$ROOT/model-paths-${SLURM_JOB_ID}.txt"
run_python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("\n".join(d.get(a,{}).get("path", "") for a in ("base","grpo","tandem","junior")))' \
    "${MODELS_JSON:?set MODELS_JSON to the pinned checkpoint paths}" > "$MODEL_PATHS"
mapfile -t MODELS < "$MODEL_PATHS"
BASE=${MODELS[0]}
GRPO=${MODELS[1]:-}
TANDEM=${MODELS[2]:-}
JUNIOR=${MODELS[3]:-$BASE}
PAIR=0
if [ -n "${MODELS[3]:-}" ]; then PAIR=1; fi
if [ "$MODE" = junior-solo ] && [ "$PAIR" = 0 ]; then echo "junior-solo requires an explicit junior manifest entry" >&2; exit 2; fi
LIMIT=${LIMIT:-0}
SOLO_N=${SOLO_N:-32}
HANDOFF_N=${HANDOFF_N:-8}

# Individual phases write atomically; a retry reuses only finished phases.
ARMS=(base grpo tandem)
HANDOFF_ARMS=(grpo tandem)
if [ "$PAIR" = 1 ]; then ARMS=(base junior grpo tandem); HANDOFF_ARMS=(base grpo tandem); fi
case "$MODE" in
    base) ARMS=(base); HANDOFF_ARMS=();;
    *-solo) ARMS=("${MODE%-solo}"); HANDOFF_ARMS=();;
    *-handoff) ARMS=(); HANDOFF_ARMS=("${MODE%-handoff}");;
esac
for ARM in "${ARMS[@]}"; do
    case "$ARM" in base) MODEL=$BASE;; junior) MODEL=$JUNIOR;; grpo) MODEL=$GRPO;; tandem) MODEL=$TANDEM;; esac
    : "${MODEL:?Missing model path for $ARM}"
    OUT="$ROOT/$ARM/solo.json"
    if [ ! -f "$OUT" ]; then
        echo "START $ARM solo $(date -u +%FT%TZ)"
        run_python eval/solo.py --model "$MODEL" --out "$OUT" --n "$SOLO_N" --limit "$LIMIT"
    fi
done
for ARM in "${HANDOFF_ARMS[@]}"; do
    case "$ARM" in base) MODEL=$BASE;; grpo) MODEL=$GRPO;; tandem) MODEL=$TANDEM;; esac
    : "${MODEL:?Missing model path for $ARM}"
    OUT="$ROOT/$ARM/handoff.json"
    if [ ! -f "$OUT" ]; then
        echo "START $ARM handoff $(date -u +%FT%TZ)"
        run_python eval/handoff.py --senior "$MODEL" --junior "$JUNIOR" \
            --out "$OUT" --n "$HANDOFF_N" --limit "$LIMIT"
    fi
done
if [ "$MODE" = full ] && [ "$LIMIT" = 0 ] && [ "$SOLO_N" -ge 32 ] && [ "$HANDOFF_N" -ge 8 ]; then
    PLOT_ARGS=()
    if [ "$PAIR" = 1 ]; then PLOT_ARGS+=(--paired-baselines); fi
    run_python eval/figure2.py "${PLOT_ARGS[@]}" --results "$ROOT" --out "$ROOT/figure2"
fi
echo "DONE $(date -u +%FT%TZ)"
