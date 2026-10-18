# Tandem RLVR

**[Final report (PDF)](final_report.pdf)** · **[Model weights on Hugging Face](https://huggingface.co/cge7-tandem-rlvr/tandem-rlvr)**

Tandem RLVR trains a senior LLM with GRPO while a frozen junior model writes
alternating parts of each rollout. This
repository reproduces the method on Qwen3-4B and extends it to a Qwen3.5-9B
senior with a Qwen3.5-2B junior, to a coding dataset, and to a synthesized math
dataset testing whether Tandem RLVR changes the model's tendency to try highly
elegant versus bashy solutions. Further details are in the
[written report](final_report.pdf).

![Tandem RLVR overview](docs/teaser.png)

*Figure 1 from D. Jiao, R. Singhal, R. West, A. Anderson,
[Tandem Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2606.28166),
arXiv:2606.28166, 2026.*

## Repository layout

`*` marks code added after forking CSSLab/Tandem-RLVR.

```
data/                 DeepScaleR builder, eval benchmarks
reward/               Math answer grader
train/
  vanilla_grpo.sh     GRPO arm
  tandem_grpo.sh      Tandem arm
eval/
  solo.py             Solo pass@k
  handoff.py          Senior/junior handoff
  legibility.py       Junior cross-entropy
  select_figure2.py * Checkpoint selection
  regrade.py        * Regrade saved answers
  figure2.py        * Figure 2 plot
  figure3.py        * Token-drift analysis
experiments/        *
  coding/             APPS coding
  method_audit/       Builds elegant vs. bashy solution dataset
  elegance/           Elegance-rewarded RL
  shorthand/          Failed attempts at inducing jargon via GRPO
third_party/          verl/vLLM Tandem patches
slurm/              * Setup and cluster jobs
```

## Rerunning the experiments

We used 80 GB A100/H100 GPUs for the 4B runs and H200s for the 9B and coding
runs; each training run took roughly 30–60 GPU-hours.

**1. Install**:

```bash
export REPO="$PWD" RUN_ROOT="$PWD/runs" UV_BIN_DIR="$(dirname "$(command -v uv)")"
TANDEM_ENV="$PWD/.envs/training"  PROFILE=training  PREPARE_FORKS=1 bash slurm/setup.sh
TANDEM_ENV="$PWD/.envs/inference" PROFILE=inference bash slurm/setup.sh
```

**2. Prepare data** (DeepScaleR: 39,309 train / 1,000 validation rows):

```bash
uv run --no-project --python .envs/training/bin/python python data/build_deepscaler.py --out-dir data/deepscaler
```

**3. Train both arms** (200 steps; checkpoints in `checkpoints/<run>/hf/global_step_N`):

```bash
cp train/env.sh.example train/env.sh   # set TANDEM_ENV_BIN=$PWD/.envs/training/bin
GPUS=0,1 TOTAL_STEPS=200 bash train/vanilla_grpo.sh
GPUS=0,1 TOTAL_STEPS=200 bash train/tandem_grpo.sh
```

**4. Evaluate** solo pass@k and paragraph handoff to the frozen junior on AMC, AIME, Minerva and OlympiadBench:

```bash
export PATH="$PWD/.envs/inference/bin:$PATH" JUNIOR=Qwen/Qwen3-4B-Instruct-2507
declare -A CKPT=(
  [grpo]=checkpoints/vanilla_grpo_qwen3_4b_deepscaler/hf/global_step_160
  [tandem]=checkpoints/tandem_grpo_qwen3_4b_deepscaler/hf/global_step_180
)
python eval/solo.py --model $JUNIOR --out results/figure2/base/solo.json --n 32
for arm in grpo tandem; do
  python eval/solo.py --model ${CKPT[$arm]} --out results/figure2/$arm/solo.json --n 32
  python eval/handoff.py --senior ${CKPT[$arm]} --junior $JUNIOR --out results/figure2/$arm/handoff.json --n 8
done
```

**5. Plot:**

```bash
uv run --no-project --python .envs/inference/bin/python python eval/figure2.py \
  --results results/figure2 --out results/figure2/figure2
```

### Extensions

Each extension trains on one GPU:

```bash
export REPO=$PWD UV_BIN_DIR=$(dirname $(command -v uv)) GPUS=0 CUDA_VISIBLE_DEVICES=0
TRAIN=$PWD/.envs/training INFER=$PWD/.envs/inference
PY="uv run --no-project --python $TRAIN/bin/python python"
```

**Qwen3.5-9B senior / Qwen3.5-2B junior:**

```bash
R=$PWD/runs/qwen35
$PY eval/download_figure2.py --manifest eval/qwen35-models.json --out $R/base.json
$PY train/assets/build_word_boundary_ids.py --model $(jq -r .base.path $R/base.json) --out $R/word-boundaries.json --write
for A in grpo tandem; do
  bash slurm/train-figure2.sbatch RUN_ROOT=$R DATA_ROOT=$PWD/data TANDEM_ENV=$TRAIN ARM=$A MODE=full
done
$PY eval/select_figure2.py --root $R --out $R/models.json
bash slurm/figure2.sbatch TANDEM_ENV=$INFER MODELS_JSON=$R/models.json RESULTS_ROOT=$R/results
```

**APPS coding** (needs `bwrap` for the test sandbox):

```bash
export RUN_ROOT=$PWD/runs/apps DATA_ROOT=$PWD/runs/apps/data TANDEM_ENV=$TRAIN
J=experiments/coding/job.sbatch
$PY eval/download_figure2.py --out $RUN_ROOT/base.json
DATA_REVISION=21e74ddf8de1a21436da12e3e653065c5213e9d1 bash $J setup
for P in solo word-handoff; do EVAL_PHASE=$P bash $J calibration; done; bash $J gate
for A in grpo tandem; do ARM=$A MODE=full bash $J train; done
bash $J select
for P in solo legibility; do for A in base grpo tandem; do ARM=$A EVAL_PHASE=$P bash $J evaluate; done; done
for P in handoff word-handoff; do for A in grpo tandem; do ARM=$A EVAL_PHASE=$P bash $J evaluate; done; done
bash $J plot
```

**Elegance RL** (uses the 9B/2B models from above). The elegance reward is
judged by GPT-6.1-sol through the Codex CLI, so run `codex login` first:

```bash
E=$PWD/runs/elegance
uv run --no-project --python $INFER/bin/python python -m experiments.elegance.prepare \
  --archive data/method-elegance/dataset.zip --out $E/data
for A in grpo tandem; do
  bash experiments/elegance/job.sbatch RUN_ROOT=$E ARM=$A TANDEM_ENV=$TRAIN INITIALIZATION_ROOT=$R \
    ELEGANCE_DATA=$E/data ELEGANCE_CODEX_BIN=$(command -v codex)
done
uv run --no-project --with matplotlib,pandas python experiments/elegance/figure/plot_elegance.py
```
