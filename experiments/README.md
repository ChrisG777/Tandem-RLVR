Additional experiments reuse the reproduction trainer and utilities without changing the Figure 2 workflow.

## Coding

[APPS preparation](coding/build_apps.py) → [calibration and evaluation](coding/campaign.py)
→ [GRPO/Tandem campaign](coding/submit.py), using a [sandboxed executable reward](coding/reward.py).
[Protocol and interfaces](../docs/APPS.md).

## MedCalc

[Prepare verified calculations](medcalc/build_data.py) → [train](medcalc/train.sh)
→ [evaluate](medcalc/evaluate.sh) → [report](medcalc/report.py).
[Protocol, environment, and interfaces](medcalc/README.md).

## Method Atlas and elegance

[Curate pairs](method_audit/curate.py) → [review](method_audit/review.py)
→ [verify answers](method_audit/verify.py) → [label elegance](method_audit/elegance.py)
→ [package](method_audit/package.py) / [inspect](method_audit/serve.py).
[Records and protocol](../docs/METHOD_AUDIT.md).

[Prepare fixed holdouts](elegance/prepare.py) → [judged reward](elegance/reward.py)
→ [solo/Tandem training](elegance/job.sbatch). [Elegance RL protocol](../docs/ELEGANCE_RL.md).

## Shorthand and concepts

[Build matrix/string splits](shorthand/build_shorthand.py) or
[build RuleTaker/Re-ARC splits](shorthand/build_concept_pilot.py) →
[calibrate](shorthand/slurm/shorthand-eval.sbatch) → [submit](shorthand/submit.sh)
→ train → verify → evaluate → [blinded review](shorthand/review_shorthand.py).

| Interface | Contract |
|---|---|
| [Dataset builders](shorthand/build_shorthand.py) | Pinned generators and seed → disjoint hashed splits; [uv project](shorthand/env/pyproject.toml) supplies generator dependencies |
| [Correctness reward](shorthand/shorthand_reward.py) | Task, response, target → score and independent accuracy/format diagnostics |
| [Length adapter](shorthand/length_penalty.py) | Correctness and unpadded token count → optional penalized reward |
| [Checks](shorthand/check_shorthand.py) | Calibration/training artifacts → verification receipt or failure |
| [Evaluate](shorthand/shorthand.py) | Model, splits, token budgets → resumable traces, accuracy, lengths, truncation |
| [Submit](shorthand/submit.sh) | Tasks, two budgets, calibration job, explicit site resources → guarded 100-step campaign |

[Submission interface and calibration probes](../docs/WORKFLOWS.md#pilot-submission).
Dataset artifacts remain in `data/shorthand/` and `data/concept-pilot/`; moving code does not change their hashes.
[Protocol](../docs/SHORTHAND_EXPERIMENT.md) · [Concepts](../docs/CONCEPT_PILOTS.md) · [Length reward](../docs/LENGTH_PENALTY_PILOTS.md).

## Natural-language drift

[Prepare GSM8K/base weights](drift/build_drift.py) →
[train](drift/slurm/drift-train.sbatch) → [select and evaluate](drift/drift.py).
[Reward](drift/gsm8k_drift.py) is final numeric correctness; first-answer accuracy is diagnostic.
The [evaluation job](drift/slurm/drift-eval.sbatch) saves resumable base/selected traces.
[Protocol and limitations](../docs/NATURAL_DRIFT_PILOT.md).

## Native word handoff

[Evaluate](word_handoff/word_handoff.py) or use its [job](word_handoff/slurm/word-handoff.sbatch):
selected senior plus frozen junior → native-sampler traces, per-token authorship, and scores.
This uses the existing training sampler, unlike Figure 2's paragraph-level evaluator.
Training/evaluation decoding presets remain separate. [Protocol](../docs/FIGURE2.md#word-level-training-schedule-evaluation).
