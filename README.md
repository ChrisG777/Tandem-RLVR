Reproduce Tandem RLVR and test whether solo RL develops reusable, opaque shorthand on repeated reasoning tasks.

## Core records

| Record | Meaning |
|---|---|
| [Dataset manifest](data/build_shorthand.py) | Generator revision, task configuration, split seeds, and data hashes |
| [Verified checkpoint](train/figure2_checkpoint.py) | Selected weights, held-out score, and explicit gaps in observed training metrics |
| [Evaluation manifest](eval/select_figure2.py) | Official base and independently trained GRPO/Tandem checkpoint paths |
| [Pilot evaluation](eval/shorthand.py) | Solo generations, correctness, lengths, and truncation for one task/budget |

## Figure 2 reproduction

[Prepare DeepScaleR](data/build_deepscaler.py) → [submit guarded training](slurm/submit-training.sh)
→ [verify/select checkpoints](eval/select_figure2.py) → [evaluate and plot](slurm/figure2.sbatch).

| Entry point | Essential inputs → outputs |
|---|---|
| [Training job](slurm/train-figure2.sbatch) | Base/data paths, arm, smoke/full mode → checkpoints and validation logs |
| [Tandem](train/tandem_grpo.sh) / [solo GRPO](train/vanilla_grpo.sh) | Model/data configuration → policy updates; Tandem trains only senior-authored tokens |
| [Checkpoint verification](train/figure2_checkpoint.py) | Run directory, arm, step count → best verified held-out checkpoint; resume evidence records missing metrics explicitly |
| [Training progress](eval/training_progress.py) | Campaign directory or snapshot → reward/validation plots and health summary |
| [Solo](eval/solo.py), [handoff](eval/handoff.py), [legibility](eval/legibility.py) | Model(s), benchmark configuration → accuracy or junior cross-entropy |
| [Figure 2](eval/figure2.py) | Completed solo/handoff evaluations → figure and bootstrap statistics |

[Training protocol/setup](docs/FRESH_TRAINING.md) · [Evaluation protocol](docs/FIGURE2.md)
· [Monitoring](docs/TRAINING_PROGRESS.md) · [Run history](docs/FIGURE2_RUN.md)

## Solo shorthand pilot

Reasoning Gym matrix transformations and string rewriting at 256- and 1,024-token
budgets test whether useful opaque conventions emerge before adding Tandem arms.

| Entry point | Essential inputs → outputs |
|---|---|
| [Build dataset](data/build_shorthand.py) | Pinned generator/configuration/seed → disjoint Parquet splits and manifest |
| [Reward](reward/shorthand_reward.py) | Task/response/target → binary correctness and format diagnostics |
| [Submit pilot](slurm/submit-shorthand.sh) | CSAIL paths/calibration job → four solo runs, evaluations, and review artifacts |
| [Calibration/training checks](train/check_shorthand.py) | Calibration or training artifacts → verified gate status |
| [Evaluate](eval/shorthand.py) | Checkpoint/test split/budget → solo traces and metrics |
| [Prepare review](eval/review_shorthand.py) | Completed evaluations → paired metrics, blinded traces, and authorship key |

[Protocol, data, and setup](docs/SHORTHAND_EXPERIMENT.md) · [Run history](docs/SHORTHAND_RUN.md)

Reasoning Gym supplies task generators/oracles; pinned vLLM and verl forks provide
rollouts and optimization. See [architecture](docs/ARCHITECTURE.md),
[fork patches](third_party/COMMIT_LOG.md), [environment](env/SETUP.md), and [tests](tests/README.md).
