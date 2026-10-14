# Experiment workflows

These variations reuse the reproduction trainer and utilities. Start with the
[results guide](../docs/RESULTS.md), [released checkpoints](../README.md#released-checkpoints)
and [workflow interfaces](../docs/WORKFLOWS.md).
Statuses summarize repository records through October 14, 2026, not live jobs.

The two featured extensions are the [asymmetric 9B/2B pair](#asymmetric-qwen35-pair)
and the [shorthand/length-pressure pilots](#shorthand-and-concepts).

## Asymmetric Qwen3.5 pair

This main extension trains a 9B senior with a frozen 2B junior to test an initially
unequal pair. The complete 1,064-problem evaluation shows higher macro paragraph-
handoff scores for Tandem and broadly similar solo scores; handoff gains appear
in AMC, AIME and Olympiad, while Minerva is slightly lower.

![Complete Qwen3.5-9B/2B Figure 2 benchmark comparison.](../docs/figures/figure2-qwen35-9b-2b.png)

Top: paragraph handoff. Bottom: solo. Bands: ±1 bootstrap standard error.
[Results](../docs/RESULTS.md#asymmetric-benchmark-figure-qwen35-9b--2b) · [Source](../docs/figures/README.md#benchmark-exports).

```mermaid
flowchart LR
    A["Qwen3.5-9B senior + 2B junior"] --> B["Compatibility and training smokes"]
    B --> C["Solo GRPO / word-level Tandem"]
    C --> D["Four solo evaluations<br/>Two paragraph handoffs"]
    D --> E["Complete 1,064-problem evaluation<br/>Higher macro handoff; similar solo"]
```

[Pinned models](../eval/qwen35-models.json) → [pair probe](../train/check_pair.py) →
[train](../slurm/train-figure2.sbatch) → [select](../eval/select_figure2.py) →
[evaluate](../slurm/figure2.sbatch). [Protocol and evidence](../docs/QWEN35_PAIR.md) ·
[Rollout defect investigation](../docs/QWEN35_COLLAPSE.md) · [Training performance](../docs/QWEN35_SPEED.md).

## Shorthand and concepts

This main extension uses solo-RL pilots to look for useful shorthand, then tests
whether explicit length pressure changes that behavior. Matrix and RuleTaker
training and evaluation are complete; strings and Re-ARC did not pass calibration.

### Matrix and string shorthand

Correctness-only solo RL tests whether repeated transformations produce useful
shorthand. The 2K-trained matrix arm improved test accuracy from 68.46% to 94.14%
at 3K evaluation; partial blinded review found no opaque shorthand, while strings failed calibration.

```mermaid
flowchart LR
    A["Matrix / string generators"] --> B["Calibration gate<br/>Matrix proceeds; strings stop"]
    B --> C["Solo RL at two response budgets"]
    C --> D["Standard and longer-chain tests"]
    D --> E["Matrix: 94.14% at 3K evaluation<br/>No opaque shorthand in partial review"]
```

[Build splits](shorthand/build_shorthand.py) → [calibrate](shorthand/slurm/shorthand-eval.sbatch) →
[submit](shorthand/submit.sh) → [train](shorthand/slurm/shorthand-train.sbatch) →
[verify](shorthand/check_shorthand.py) → [evaluate](shorthand/shorthand.py) →
[blinded review](shorthand/review_shorthand.py).
[Protocol](../docs/SHORTHAND_EXPERIMENT.md) · [Initial failed pilot](../docs/SHORTHAND_RUN.md) ·
[Matrix evidence](../docs/ENGAGING_PILOT.md) · [Partial review](../results/pilot-review/blinded-ratings-20261008.json).

### Reusable concepts

RuleTaker shared deductions and Re-ARC object tasks test reuse beyond shorter
English. RuleTaker's 4K all-four accuracy rose from 43.55% to 79.49%; Re-ARC's
truncation gate blocked RL, and useful emergent jargon is not established.

```mermaid
flowchart LR
    A["RuleTaker shared deductions"] --> B["Calibration and solo RL"]
    B --> C["RuleTaker all-four accuracy at 4K<br/>43.55% base to 79.49% trained"]
    D["Re-ARC object tasks"] --> E["Calibration fails; no RL"]
```

[Build RuleTaker/Re-ARC splits](shorthand/build_concept_pilot.py), then use the shared
pipeline above. [Concept protocol and gates](../docs/CONCEPT_PILOTS.md) ·
[Later RuleTaker results](../docs/LENGTH_PENALTY_PILOTS.md#collapse-investigation-2026-10-10).

### Length pressure

A token cost tests whether compression encourages reusable conventions, using fresh
solo runs and matched no-penalty controls. It reduced matrix accuracy from 92.38%
to 15.63% and RuleTaker from 79.49% to 62.11%; the proposed normalization ablation remains unrun.

```mermaid
flowchart LR
    A["Same base, data and training settings"] --> B["No cost / correctness minus token cost"]
    B --> C["Final-step accuracy, length and truncation"]
    C --> D["Accuracy falls with token cost<br/>Matrix: 92.38% to 15.63%<br/>RuleTaker: 79.49% to 62.11%"]
```

[Length adapter](shorthand/length_penalty.py) · [Intervention, controls and diagnosis](../docs/LENGTH_PENALTY_PILOTS.md).

![Measured matrix and RuleTaker accuracy: base, correctness-only RL and length-penalized RL.](../docs/figures/pilot-summary.png)

Matrix uses 3K training/evaluation budgets; RuleTaker uses 4K and all-four accuracy.
[Exact values, sources and regeneration](../docs/figures/README.md#reproducible-summary-plots).

Shared interfaces distinguish correctness, compression and semantic evidence:

| Interface | Contract |
| --- | --- |
| [Dataset builders](shorthand/build_shorthand.py) | Pinned generators and seed → disjoint hashed splits; [uv project](shorthand/env/pyproject.toml) supplies dependencies |
| [Correctness reward](shorthand/shorthand_reward.py) | Task, response, target → score and independent accuracy/format diagnostics |
| [Length adapter](shorthand/length_penalty.py) | Correctness and unpadded token count → optional penalized reward |
| [Checks](shorthand/check_shorthand.py) | Calibration/training artifacts → verification receipt or failure |
| [Evaluate](shorthand/shorthand.py) | Model, splits, token budgets → resumable traces, accuracy, lengths, truncation |
| [Submit](shorthand/submit.sh) | Tasks, two budgets, calibration job, explicit resources → guarded 100-step campaign |

[Submission interface and calibration probes](../docs/WORKFLOWS.md#pilot-submission).
Dataset artifacts remain in `data/shorthand/` and `data/concept-pilot/`; moving code does not change their hashes.

## Other studies and pipelines

| Study | Type | Recorded status |
| --- | --- | --- |
| [APPS coding](#coding) | Domain-transfer experiment | Step-200 policies released; final aggregate evaluation pending. |
| [MedCalc](#medcalc) | Implemented pipeline | No empirical comparison reported. |
| [Method Atlas](#method-atlas-and-elegance) | Dataset and audit | 374 answer-checked problems and 1,840 labeled paths released. |
| [Elegance RL](#elegance-rl) | Reward/generalization experiment | Training and recovery recorded; final comparison pending. |
| [Natural-language drift](#natural-language-drift) | Exploratory pilot | Submissions/access limits recorded; no final results reported. |
| [Native word handoff](#native-word-handoff) | Interim evaluation and case study | Partial matched results; controlled recovery experiment proposed. |

## Coding

APPS tests whether Tandem's handoff behavior extends to executable Python solutions.
Both 200-step policies are released; the latest record describes sharded final
evaluation and grader recovery, without a final aggregate comparison.

```mermaid
flowchart LR
    A["APPS introductory / interview"] --> B["Calibrate sandboxed reward"]
    B --> C["Solo GRPO / word-level Tandem"]
    C --> D["Solo, paragraph and word handoff<br/>Junior cross-entropy"]
    D --> E["Both step-200 policies released<br/>Final comparison not reported"]
```

[Prepare](coding/build_apps.py) → [calibrate and evaluate](coding/campaign.py) →
[submit campaign](coding/submit.py), using the [executable reward](coding/reward.py).
[Protocol](../docs/APPS.md) · [Run evidence](../docs/APPS_RUN.md) · [Checkpoint migration](../docs/APPS_MIGRATION.md).

## MedCalc

MedCalc transfers the comparison to verified clinical calculations with numeric,
date and duration checks; the pipeline is implemented, but no empirical comparison
is reported. The official test split is reserved for evaluation.

```mermaid
flowchart LR
    A["Verified calculation notes"] --> B["Calculator reward"]
    B --> C["Solo GRPO / word-level Tandem"]
    C --> D["Accuracy, handoff and junior CE pipeline<br/>Implemented; no results reported"]
```

[Prepare](medcalc/build_data.py) → [train](medcalc/train.sh) →
[evaluate](medcalc/evaluate.sh) → [report](medcalc/report.py).
[Protocol, environment and interfaces](medcalc/README.md).

## Method Atlas and elegance

### Audited method dataset

Method Atlas collects structural shortcuts and elementary alternatives for method
holdouts. The release has 374 answer-checked problems and 1,840 labeled paths;
elegance labels are model judgments, not measured weaker-model accessibility.

```mermaid
flowchart LR
    A["Source problems and all routes"] --> B["Review mathematics and check answers"]
    B --> C["Label elegance on every supplied path"]
    C --> D["Released: 374 problems / 1,840 paths<br/>38 conservative method holdouts"]
```

[Curate](method_audit/curate.py) → [review](method_audit/review.py) →
[verify answers](method_audit/verify.py) → [label elegance](method_audit/elegance.py) →
[package](method_audit/package.py) / [inspect](method_audit/serve.py).
[Audit protocol](../docs/METHOD_AUDIT.md) · [Method-family boundaries](../docs/METHOD_FAMILIES_AUDIT.md) ·
[Dataset and limitations](../data/method-elegance/README.md).

### Elegance RL

This tests correctness plus a bonus for valid, elegant derivations using the
9B/2B pair, with complement counting, Vieta and power of a point held out.
The recorded runs reached recovery checkpoints; no final comparison is reported.

```mermaid
flowchart LR
    A["152 train / 20 validation / 17 test"] --> B["Math correctness + judged elegance bonus"]
    B --> C["Solo GRPO / word-level Tandem"]
    C --> D["Held-out-method evaluation planned<br/>No final comparison reported"]
```

[Prepare fixed holdouts](elegance/prepare.py) → [judged reward](elegance/reward.py) →
[solo/Tandem training](elegance/job.sbatch). [Protocol and run evidence](../docs/ELEGANCE_RL.md).

## Natural-language drift

This literature-informed GSM8K pilot tests drift under correctness-only solo RL
from base models. Qwen runs were submitted; Llama/Gemma access was blocked in the
recorded checks, and no final drift or accuracy results are reported.

```mermaid
flowchart LR
    A["GSM8K and pinned base models"] --> B["Correctness-only solo RL"]
    B --> C["Select by held-out greedy accuracy"]
    C --> D["First / final answer accuracy and trace review<br/>No final results reported"]
```

[Prepare](drift/build_drift.py) → [train](drift/slurm/drift-train.sbatch) →
[select and evaluate](drift/drift.py). [Reward](drift/gsm8k_drift.py) is final numeric
correctness; first-answer accuracy is diagnostic. The [evaluation job](drift/slurm/drift-eval.sbatch)
saves resumable base/selected traces. [Protocol and limitations](../docs/NATURAL_DRIFT_PILOT.md).

## Native word handoff

This evaluates the training sampler's word-level schedule, alongside Figure 2's
paragraph handoffs. A selected partial-snapshot case study found errors from both
partners; it does not establish a Tandem recovery advantage, and fixed-prefix repair is proposed only.

```mermaid
flowchart LR
    A["Selected senior + frozen junior"] --> B["Native word-level sampling"]
    B --> C["Grades, traces and token authorship"]
    C --> D["Selected partial-snapshot case study<br/>Errors from both; no recovery advantage established"]
    D -.-> E["Proposed matched-prefix repair test"]
```

[Evaluate](word_handoff/word_handoff.py) or use its [job](word_handoff/slurm/word-handoff.sbatch).
Training/evaluation decoding presets remain separate. [Protocol](../docs/FIGURE2.md#word-level-training-schedule-evaluation) ·
[Execution and profiling](../docs/WORD_HANDOFF_EXECUTION.md) · [Error attribution and recovery proposal](../docs/WORD_HANDOFF_CASE_STUDY.md).
