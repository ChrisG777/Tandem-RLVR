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
set -euo pipefail
# Explicit KEY=VALUE arguments also support sbatch --export=NIL on CSAIL.
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?set REPO to the git checkout}"
if [ -n "${APPTAINER_IMAGE:-}" ] && [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ]; then
    export CODE_COMMIT=$(git rev-parse HEAD)
    export PATH="${APPTAINER_BIN_DIR:-/orcd/software/core/001/pkg/apptainer/1.4.2/bin}:${PATH:-/usr/bin:/bin}"
    export TANDEM_CONTAINER_ACTIVE=1
    exec apptainer exec --nv --bind "${CONTAINER_BIND:?}" "$APPTAINER_IMAGE" \
        bash "$REPO/slurm/figure2.sbatch"
fi
if [ "${TANDEM_CONTAINER_ACTIVE:-0}" = 1 ]; then export CC=/usr/bin/gcc CXX=/usr/bin/g++; fi
[ "${SLURM_RESTART_COUNT:-0}" -le "${MAX_RESTARTS:-12}" ] || { echo 'Preemption retry limit reached'; exit 1; }
export PATH="${UV_BIN_DIR:-${HOME:-/tmp}/.local/bin}:/data/scratch/${SLURM_JOB_USER:-${USER:-}}/.local/bin:${PATH:-/usr/bin:/bin}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tandem-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/tandem-cache-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer-workspace"
mkdir -p "$XDG_CACHE_HOME"
unset VLLM_TANDEM_CONFIG VLLM_TANDEM_ALL_GPUS
TANDEM_ENV=${TANDEM_ENV:-$REPO/.venv}
run_python() { uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python "$@"; }
ROOT=${RESULTS_ROOT:-results/figure2}
mkdir -p "$ROOT"
if [ -n "${CODE_COMMIT:-}" ]; then
    echo "$CODE_COMMIT" > "$ROOT/code-commit.txt"
else
    git rev-parse HEAD > "$ROOT/code-commit.txt"
fi
nvidia-smi > "$ROOT/gpus-${SLURM_JOB_ID}.txt"
MODE=${MODE:-full}
case "$MODE" in base|full|grpo-solo|tandem-solo|grpo-handoff|tandem-handoff) ;; *) echo 'Unknown evaluation mode' >&2; exit 2;; esac
run_python -c 'import json,sys; d=json.load(open(sys.argv[1])); arms=("base",) if sys.argv[2]=="base" else ("base","grpo","tandem"); print("\n".join(d[a]["path"] for a in arms))' \
    "${MODELS_JSON:?set MODELS_JSON to the pinned checkpoint paths}" "$MODE" > "$ROOT/model-paths.txt"
mapfile -t MODELS < "$ROOT/model-paths.txt"
BASE=${MODELS[0]}
GRPO=${MODELS[1]:-}
TANDEM=${MODELS[2]:-}
LIMIT=${LIMIT:-0}
SOLO_N=${SOLO_N:-32}
HANDOFF_N=${HANDOFF_N:-8}

# Individual phases write atomically; a retry reuses only finished phases.
ARMS=(base grpo tandem)
HANDOFF_ARMS=(grpo tandem)
case "$MODE" in
    base) ARMS=(base); HANDOFF_ARMS=();;
    *-solo) ARMS=("${MODE%-solo}"); HANDOFF_ARMS=();;
    *-handoff) ARMS=(); HANDOFF_ARMS=("${MODE%-handoff}");;
esac
for ARM in "${ARMS[@]}"; do
    case "$ARM" in base) MODEL=$BASE;; grpo) MODEL=$GRPO;; tandem) MODEL=$TANDEM;; esac
    OUT="$ROOT/$ARM/solo.json"
    if [ ! -f "$OUT" ]; then
        echo "START $ARM solo $(date -u +%FT%TZ)"
        run_python eval/solo.py --model "$MODEL" --out "$OUT" --n "$SOLO_N" --limit "$LIMIT"
    fi
done
for ARM in "${HANDOFF_ARMS[@]}"; do
    case "$ARM" in grpo) MODEL=$GRPO;; tandem) MODEL=$TANDEM;; esac
    OUT="$ROOT/$ARM/handoff.json"
    if [ ! -f "$OUT" ]; then
        echo "START $ARM handoff $(date -u +%FT%TZ)"
        run_python eval/handoff.py --senior "$MODEL" --junior "$BASE" \
            --out "$OUT" --n "$HANDOFF_N" --limit "$LIMIT"
    fi
done
if [ "$MODE" = full ] && [ "$LIMIT" = 0 ] && [ "$SOLO_N" -ge 32 ] && [ "$HANDOFF_N" -ge 8 ]; then
    run_python eval/figure2.py --results "$ROOT" --out "$ROOT/figure2"
fi
echo "DONE $(date -u +%FT%TZ)"
