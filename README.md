# Tandem RLVR

**[Final report (PDF)](final_report.pdf)** · **[Model weights on Hugging Face](https://huggingface.co/cge7-tandem-rlvr/tandem-rlvr)**

Tandem RLVR trains a senior LLM with GRPO while a frozen junior model writes
alternating parts of each rollout. This
repository reproduces the method on Qwen3-4B and extends it to a Qwen3.5-9B
senior with a Qwen3.5-2B junior, to a coding dataset, and to a synthesized math
dataset testing whether Tandem RLVR changes the model's tendency to try highly
technical versus simpler, bashier solutions. Further details are in the
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
  check_pair.py     * Pre-training test of 9B/2B co-generation
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
  method_audit/       Synthesized method dataset
  elegance/           Elegance-rewarded RL
  shorthand/          Shorthand pilots
third_party/          verl/vLLM Tandem patches
slurm/              * Setup and cluster jobs
docs/               * Notes and figures
```

## Rerunning the experiments

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

**4. Evaluate** solo pass@k (32 samples) and paragraph handoff to the frozen junior (8 chains) on AMC, AIME, Minerva and OlympiadBench:

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
