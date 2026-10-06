# Tandem Reinforcement Learning with Verifiable Rewards

[![arXiv](https://img.shields.io/badge/arXiv-2606.28166-b31b1b.svg)](https://arxiv.org/pdf/2606.28166) [![🤗 TT-Word-Qwen3-4B](https://img.shields.io/badge/🤗-TT--Word--Qwen3--4B--DeepScaleR-yellow)](https://huggingface.co/difanjiao/TT-Word-Qwen3-4B-Instruct-2507-DeepScaleR)

RLVR raises a model's reasoning ability without any pressure to keep that reasoning legible to the
weaker models and people it has to work with. TRLVR changes one thing in the rollout: every rollout
is co-generated with a frozen junior, a copy of the senior's own pre-RL base, and the policy
gradient covers only the tokens the senior wrote. On competition math, the Tandem RLVR senior
matches GRPO in solo reasoning capability, retains nearly all of that capability when the junior
takes over half the reasoning, and expresses its reasoning in language the junior can follow.

![Tandem RLVR](docs/teaser.png)

## Install

Tandem rollout is a patch to vLLM and verl. `third_party/` holds the pinned upstream commits and
the two patches.

```bash
bash env/install.sh $HOME/envs/tandem-rlvr     # conda env and the core stack
bash third_party/apply_patches.sh              # clone both upstreams at the pins, apply the patches
```

Then build both forks into that environment, following `docs/INSTALL.md`, and check:

```bash
python env/smoke.py
```

## Train

```bash
cp train/env.sh.example train/env.sh           # then set the paths for your machine
python data/build_deepscaler.py                # writes train.parquet and heldout.parquet
```

```bash
bash train/tandem_grpo.sh                      # tandem rollout, senior only gradient
bash train/vanilla_grpo.sh                     # matched control
bash train/kl_reg.sh                           # ablation, KL_COEF required
bash train/grpo_g16.sh                         # ablation, control at 16 rollouts
```

The standard launchers default to two GPUs. The reproduction's batch environment
uses one GPU for GRPO and two for Tandem (trainable senior plus frozen junior).
A checkpoint is saved every 20 steps, and the
Hugging Face weights of every save are kept under `${CKPT_ROOT}/${EXP_NAME}/hf/global_step_N`.
Training validates on `data/deepscaler/heldout.parquet` and logs
`val-core/deepscaler/acc/best@4/mean`; take the step with the highest value. Details in
`docs/REPRODUCING.md`.

## Evaluate

Three metrics, one script each, one JSON each.

| script | measures |
|---|---|
| `eval/solo.py` | the senior alone: pass@k over n independent samples per problem |
| `eval/handoff.py` | senior and junior alternating at every `\n\n`, one reasoning step each, under a shared token budget, with the team's answer graded |
| `eval/legibility.py` | the frozen junior's mean per-token cross-entropy, in nats, over the senior's chain of thought |

```bash
MODEL=$SENIOR TAG=step_120 RESULTS_ROOT=results/tandem bash eval/run_all.sh
```

Sampling is defined once, in `eval/config.py`. See `eval/README.md`.

[`eval/figure2.py`](eval/figure2.py) turns the three solo and two handoff results
into Figure 2 with bootstrap standard-error bands. The reproduction trains both
arms from the official Qwen base; see [docs/FRESH_TRAINING.md](docs/FRESH_TRAINING.md)
and [docs/FIGURE2.md](docs/FIGURE2.md).

| Reproduction interface | Inputs → outputs |
|---|---|
| [`train-figure2.sbatch`](slurm/train-figure2.sbatch) | `ARM=grpo/tandem`, `MODE=smoke/full`, pinned base and data → checkpoints and validation logs |
| [`verify_training`](train/figure2_checkpoint.py) | run directory, arm, step count → verified best held-out checkpoint |
| [`check_ray.py`](env/check_ray.py) | sourced batch environment → verified Ray worker interpreter and fork paths |
| [`select_figure2.py`](eval/select_figure2.py) | completed training campaign → `MODELS_JSON` for evaluation |
| [`submit-training.sh`](slurm/submit-training.sh) | CSAIL paths, optional setup job and `ARMS` → speculative full/smoke pairs and cancellation monitors |
