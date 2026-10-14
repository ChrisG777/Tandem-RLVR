Reproduce Tandem RLVR: train a senior language model with a frozen junior partner and compare it with solo GRPO across reasoning tasks.

## Data and core records

| Record | Meaning |
| --- | --- |
| [Dataset manifest](data/build_deepscaler.py) | Pinned source, disjoint splits, seed, and file hashes. |
| [Model manifest](eval/select_figure2.py) | Initial senior, optional distinct junior, and selected GRPO/Tandem checkpoints. |
| [Verified checkpoint](train/figure2_checkpoint.py) | Validation-selected weights, integrity checks, and explicit gaps in training evidence. |
| [Evaluation result](eval/common.py) | Model/data identity, saved generations, scores, and resumable progress. |

## Training

[build_deepscaler.py](data/build_deepscaler.py) prepares training and held-out data.
[submit-training.sh](slurm/submit-training.sh) composes smoke checks and full training;
[train-figure2.sbatch](slurm/train-figure2.sbatch) runs one arm:

```text
base weights + training split
→ solo GRPO: senior generates every token
  or Tandem: senior + frozen junior alternate at word boundaries
→ task reward → verl GRPO updates the senior's full weights
→ validation + saved state → verified checkpoint → model manifest
```

[vanilla_grpo.sh](train/vanilla_grpo.sh) and [tandem_grpo.sh](train/tandem_grpo.sh)
configure the pinned verl/vLLM forks. Tandem applies the policy loss only at
senior-authored positions; junior tokens remain context and junior weights stay frozen.
[verify_training](train/figure2_checkpoint.py) checks training evidence and weights;
[select_figure2.py](eval/select_figure2.py) writes the evaluation manifest.

The [Qwen3.5 pair](docs/QWEN35_PAIR.md) uses a distinct senior and junior through
the same interfaces, with a [native compatibility probe](train/check_pair.py).
[Execution options](train/execution.sh) compose with existing experiment launchers.
[Checkpoint storage](docs/ARCHITECTURE.md#other-fork-changes) separates
recovery saves from evaluation exports through `PERSIST_HF_INTERVAL`.

## Evaluation

[figure2.sbatch](slurm/figure2.sbatch) takes a model manifest and benchmark selection:

```text
initial / trained weights → solo → accuracy
trained senior + frozen junior → paragraph handoff → accuracy
senior traces + junior → legibility → cross-entropy
saved solo / handoff results → aggregate + bootstrap → Figure 2
```

[solo.py](eval/solo.py), [handoff.py](eval/handoff.py), and
[legibility.py](eval/legibility.py) save scores and traces.
[figure2.py](eval/figure2.py) aggregates the benchmark results;
[regrade.py](eval/regrade.py) rescores saved generations while retaining provenance.
Training uses native word-level switching; Figure 2 handoff evaluation uses paragraphs.

## Inspect results

[training_progress.py](eval/training_progress.py) turns logs into curves and diagnostics.
[comparison_browser.py](eval/comparison_browser.py) serves saved evaluations in the
[trace UI](ui/comparison/). [figure3.py](eval/figure3.py) measures token drift and
Jensen–Shannon divergence from saved solo outputs.

## Experiment workflows

Each experiment reuses the trainer and supplies its own data and reward:

| Workflow | Inputs → outputs |
| --- | --- |
| [APPS coding](docs/APPS.md) | Coding problems → sandboxed reward → matched GRPO/Tandem evaluation; [migration recovery](docs/APPS_MIGRATION.md). |
| [MedCalc](experiments/medcalc/README.md) | Verified calculation problems → numeric reward → matched training and evaluation. |
| [Method Atlas](docs/METHOD_AUDIT.md) | Curated method pairs → reviews and executable checks → labeled dataset and audit UI. |
| [Elegance RL](docs/ELEGANCE_RL.md) | Fixed method holdout → judged reward → matched solo/Tandem training. |
| [Shorthand and concepts](experiments/README.md#shorthand-and-concepts) | Generated tasks → correctness/length reward → traces and blinded review. |
| [Natural-language drift](experiments/README.md#natural-language-drift) | GSM8K → correctness-only training → selected checkpoint and traces. |
| [Native word handoff](experiments/README.md#native-word-handoff) | Senior/junior pair → training-sampler traces and per-token authorship; [execution options](docs/WORD_HANDOFF_EXECUTION.md). |

See [setup and job interfaces](docs/WORKFLOWS.md), [training protocol](docs/FRESH_TRAINING.md),
[evaluation protocol](docs/FIGURE2.md), and [grading audit](docs/REPRODUCTION_AUDIT.md)
for operational details. [Architecture](docs/ARCHITECTURE.md),
[fork patches](third_party/COMMIT_LOG.md), and [tests](tests/README.md) describe the implementation boundaries.
