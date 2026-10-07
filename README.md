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
| [Walltime continuation](slurm/continue-training.py) | Existing job IDs/checkpoint roots → bounded same-job requeue before expiry; resumes through the training launcher |
| [Solo](eval/solo.py), [handoff](eval/handoff.py), [legibility](eval/legibility.py) | Model(s), benchmark configuration → accuracy or junior cross-entropy |
| [Evaluation job](slurm/figure2.sbatch) | Pinned model manifest, `MODE=base` or `full` → frozen-base solo evaluation or all five evaluations and figure |
| [Figure 2](eval/figure2.py) | Completed solo/handoff evaluations → figure and bootstrap statistics |

[Training protocol/setup](docs/FRESH_TRAINING.md) · [Evaluation protocol](docs/FIGURE2.md)
· [Monitoring](docs/TRAINING_PROGRESS.md) · [Run history](docs/FIGURE2_RUN.md)

## Solo shorthand pilot

Reasoning Gym tasks test whether useful opaque conventions emerge before adding
Tandem arms. The calibrated pilot uses matrix transformations at two token budgets.

| Entry point | Essential inputs → outputs |
|---|---|
| [Build dataset](data/build_shorthand.py) | Pinned generator/configuration/seed → disjoint Parquet splits and manifest |
| [Reward](reward/shorthand_reward.py) | Task/response/target → binary correctness and format diagnostics |
| [Submit Engaging pilot](slurm/submit-engaging-pilot.sh) | Prepared environments/calibration jobs/site paths → two solo runs, CPU checks, evaluations, and review |
| [Calibration/training checks](train/check_shorthand.py) | Calibration or training artifacts and task/budget selection → verified gate status |
| [Calibration array](slurm/shorthand-calibrate.sbatch) | Site paths and task index 0–1 → independent larger-budget calibration traces ([setup](slurm/shorthand-calibration-setup.sh)) |
| [Evaluate](eval/shorthand.py) | Checkpoint/test split/budget → solo traces and metrics |
| [Prepare review](eval/review_shorthand.py) | Completed evaluations/expected comparison count → paired metrics, blinded traces, and authorship key |

[Engaging campaign](docs/ENGAGING_PILOT.md) · [Protocol, data, and setup](docs/SHORTHAND_EXPERIMENT.md) · [Run history](docs/SHORTHAND_RUN.md)

Reasoning Gym supplies task generators/oracles; pinned vLLM and verl forks provide
rollouts and optimization. See [architecture](docs/ARCHITECTURE.md),
[fork patches](third_party/COMMIT_LOG.md), [environment](env/SETUP.md), and [tests](tests/README.md).
