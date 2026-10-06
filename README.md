# Tandem RLVR: reusable shorthand

Test whether solo RL develops useful but opaque shorthand on repeated reasoning tasks, before testing whether tandem RLVR prevents it.

## Pilot

Four **solo GRPO** runs: [Reasoning Gym](https://github.com/open-thought/reasoning-gym) matrix transformations and string rewriting, each with 256- and 1,024-token responses. Qwen3-4B-Instruct-2507; 100 updates, batch 16, eight rollouts/prompt, LR 1e-6, seed 42. Reward is final-answer correctness only; no supplied shorthand. Tandem runs are deferred.

Each task has 4,096 train, 128 validation, 256 test, 256 longer-chain test, and 64 calibration examples. Base and trained policies are evaluated alone at both budgets. Calibration gates training; matched, blinded trace pairs support checking whether notation is new, reusable, and difficult to understand. Shorter output alone is not evidence of jargon.

## Interfaces

| Interface | Inputs → outputs |
|---|---|
| [build_dataset](data/build_shorthand.py) | Pinned generator/config/seed → disjoint Parquet splits and manifest |
| [compute_score](reward/shorthand_reward.py) | Task/response/target → binary correctness and answer-format diagnostics |
| [Submit pilot](slurm/submit-shorthand.sh) | CSAIL paths/calibration job → four solo runs, evaluations, review artifacts |
| [evaluate](eval/shorthand.py) | Checkpoint/test split/budget → solo traces, accuracy, lengths, truncation |
| [build_review](eval/review_shorthand.py) | Completed evaluations → paired metrics, blinded trace pairs, answer key |

[Protocol, sources, and setup](docs/SHORTHAND_EXPERIMENT.md) · [Job status](docs/SHORTHAND_RUN.md) · [Tandem architecture](docs/ARCHITECTURE.md)
