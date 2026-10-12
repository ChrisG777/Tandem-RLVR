Reproduce Tandem RLVR with the authors’ training and evaluation code, alongside separate reasoning experiments.

## Core records

| Record | Meaning |
|---|---|
| [Data manifest](data/build_deepscaler.py) | Dataset revision, split seed, and file hashes |
| [Verified checkpoint](train/figure2_checkpoint.py) | Validation-selected weights and explicit gaps in observed metrics |
| [Evaluation manifest](eval/select_figure2.py) | Pinned base, optional frozen junior, and selected GRPO/Tandem paths |

## Reproduction

[Prepare DeepScaleR](data/build_deepscaler.py) → [submit training](slurm/submit-training.sh)
→ [select checkpoints](eval/select_figure2.py) → [evaluate](slurm/figure2.sbatch)
→ [aggregate and plot](eval/figure2.py).

| Entry point | Inputs → outputs |
|---|---|
| [Solo GRPO](train/vanilla_grpo.sh) / [Tandem](train/tandem_grpo.sh) | Base weights and data → policy checkpoints, using pinned verl/vLLM |
| [Solo](eval/solo.py) / [paragraph handoff](eval/handoff.py) / [legibility](eval/legibility.py) | Model(s) and benchmarks → scores and traces; [original runner](eval/run_all.sh) composes these |
| [Checkpoint verification](train/figure2_checkpoint.py) | Training logs and saved state → verified selection; [migration gate](slurm/prepare-migrated-training.sbatch) validates transferred state |
| [Regrade](eval/regrade.py) | Saved generations → corrected scores with original grades and provenance |
| [Figure 3 token drift](eval/figure3.py) | Figure 2 solo texts → marginal token drift, JSD and token case studies ([definitions](docs/FIGURE3.md)) |
| [Training progress](eval/training_progress.py) | Logs or snapshots → curves and diagnostics |
| [Trace browser](eval/comparison_browser.py) | Saved evaluations → local side-by-side browser; [UI](ui/comparison/) separate from data logic ([usage](docs/COMPARISON_BROWSER.md)) |

[Qwen3.5 asymmetric pair](docs/QWEN35_PAIR.md): pinned 9B senior / 2B junior →
[native compatibility probe](train/check_pair.py) → the same training and evaluation workflow.

## Shared job tools

| Entry point | Inputs → outputs |
|---|---|
| [Setup](slurm/setup.sh) | Training/inference/grading profile and paths → pinned environment and checks |
| [Smoke guard](slurm/watch-smoke.py) | Smoke/full job IDs and verification command → status; cancels the full job on failure |
| [Runtime](slurm/runtime.sh) | Entrypoint and checkpoint/progress location → container execution and bounded same-job requeue |

[Current setup and job interfaces](docs/WORKFLOWS.md) · [Reproduction protocol](docs/FRESH_TRAINING.md)
· [Evaluation protocol](docs/FIGURE2.md) · [Grading limitations and audit](docs/REPRODUCTION_AUDIT.md)

## Additional experiments

[Shorthand and concept pilots](experiments/README.md#shorthand-and-concepts),
[correctness-only drift](experiments/README.md#natural-language-drift), and
[native word handoff](experiments/README.md#native-word-handoff) live under `experiments/`.
They reuse the original trainer and evaluation utilities; they are not part of the Figure 2 protocol.

[Architecture](docs/ARCHITECTURE.md) · [Fork patches](third_party/COMMIT_LOG.md) · [Tests](tests/README.md)

[APPS coding](docs/APPS.md): [prepare data](experiments/coding/build_apps.py) →
[sandboxed reward](experiments/coding/reward.py) →
[calibrate, train and evaluate](experiments/coding/campaign.py).
