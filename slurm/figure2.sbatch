#!/usr/bin/env bash
# Select account/partition/QoS when submitting; both engines need one GPU each.
#SBATCH --job-name=tandem-figure2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/figure2-%j.out
#SBATCH --requeue
set -euo pipefail
# Explicit KEY=VALUE arguments also support sbatch --export=NIL on CSAIL.
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?set REPO to the git checkout}"
export PATH="$HOME/.local/bin:/data/scratch/$USER/.local/bin:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tandem-mpl-${SLURM_JOB_ID}"
unset VLLM_TANDEM_CONFIG VLLM_TANDEM_ALL_GPUS
TANDEM_ENV=${TANDEM_ENV:-$REPO/.venv}
run_python() { uv run --python "$TANDEM_ENV/bin/python" --no-project --offline "$@"; }
ROOT=${RESULTS_ROOT:-results/figure2}
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code-commit.txt"
nvidia-smi > "$ROOT/gpus-${SLURM_JOB_ID}.txt"
run_python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("\n".join(d[a]["path"] for a in ("base","grpo","tandem")))' \
    results/figure2/models.json > "$ROOT/model-paths.txt"
mapfile -t MODELS < "$ROOT/model-paths.txt"
BASE=${MODELS[0]}
GRPO=${MODELS[1]}
TANDEM=${MODELS[2]}
LIMIT=${LIMIT:-0}
SOLO_N=${SOLO_N:-32}
HANDOFF_N=${HANDOFF_N:-8}

# Individual phases write atomically; a retry reuses only finished phases.
for ARM in base grpo tandem; do
    case "$ARM" in base) MODEL=$BASE;; grpo) MODEL=$GRPO;; tandem) MODEL=$TANDEM;; esac
    OUT="$ROOT/$ARM/solo.json"
    if [ ! -f "$OUT" ]; then
        echo "START $ARM solo $(date -u +%FT%TZ)"
        run_python eval/solo.py --model "$MODEL" --out "$OUT" --n "$SOLO_N" --limit "$LIMIT"
    fi
done
for ARM in grpo tandem; do
    case "$ARM" in grpo) MODEL=$GRPO;; tandem) MODEL=$TANDEM;; esac
    OUT="$ROOT/$ARM/handoff.json"
    if [ ! -f "$OUT" ]; then
        echo "START $ARM handoff $(date -u +%FT%TZ)"
        run_python eval/handoff.py --senior "$MODEL" --junior "$BASE" \
            --out "$OUT" --n "$HANDOFF_N" --limit "$LIMIT"
    fi
done
if [ "$LIMIT" = 0 ] && [ "$SOLO_N" -ge 32 ] && [ "$HANDOFF_N" -ge 8 ]; then
    run_python eval/figure2.py --results "$ROOT" --out "$ROOT/figure2"
fi
echo "DONE $(date -u +%FT%TZ)"
