# Results at a glance

Tandem RLVR asks whether training with a frozen partner preserves solo reasoning
while making the trained model's reasoning easier for that partner to continue.
Our Qwen3-4B runs report comparable solo scores and less token drift, but **do not
reproduce the paper's clear paragraph-handoff advantage**.

This guide summarizes the experiment records available on October 14, 2026.
The two main extensions are the [asymmetric 9B/2B pair](#asymmetric-benchmark-figure-qwen35-9b--2b)
and [shorthand/length-pressure pilots](#pilot-results-accuracy-versus-length-pressure).
See the [plot index and source files](figures/README.md) and [experiment map](../experiments/README.md).

## Headline comparison: Figure 2

Both arms start from the same Qwen3-4B-Instruct-2507 base and train for 200 updates
on DeepScaleR. Validation selects GRPO step 160 and Tandem step 180. Tandem trains
with a frozen copy of the base, switching at word boundaries; only senior-authored
tokens receive policy loss. See the [training protocol](FRESH_TRAINING.md).

![Complete Qwen3-4B Figure 2 comparison across AMC, AIME, Minerva, Olympiad and the macro average.](figures/figure2-qwen3-4b.png)

Top: paragraph handoff. Bottom: solo. Bands: ±1 bootstrap standard error.

The [completed, corrected evaluation](REPRODUCTION_AUDIT.md#completion-update--2026-10-09-pacific)
covers 1,064 problems: AMC 121, AIME 90, Minerva 272, and OlympiadBench 581.
Each benchmark has equal weight in the macro average. Solo uses 32 attempts per
problem; handoff uses eight complete responses, alternating models at paragraph
breaks within a shared 3,000-token budget. `pass@k` estimates the probability of
at least one correct answer in *k* attempts.

| Macro metric | GRPO | Tandem | Tandem − GRPO |
|---|---:|---:|---:|
| Solo pass@4 | 59.46% | 59.97% | +0.51 points |
| Paragraph handoff pass@4 | 58.25% | 57.55% | −0.70 points |
| Paragraph handoff pass@8 | 61.79% | 60.48% | −1.31 points |

These small differences describe one training run per arm. They do not establish
that either method is generally better. The [evaluation protocol](FIGURE2.md)
also records the paper's comparison targets and sampling choices.

### Why the grading correction matters

The original training and evaluation environments omitted the authors' optional
math-verification dependencies, causing false negatives. Restoring those packages
and correcting scientific-notation parsing largely removed the apparent Minerva
deficit: corrected solo pass@4 is 56.04% for GRPO and 56.95% for Tandem.
**Regrading fixes reported scores, not the earlier policy updates or checkpoint
selection.** The selected checkpoints still reflect the original rewards, and
Tandem validation observations at steps 20 and 160 are missing. Corrected absolute
scores should not be equated with the paper's scores without matching its runtime.
See the [grading audit and limitations](REPRODUCTION_AUDIT.md#confirmed-grading-problem).

## Token drift: Figure 3

Across 34,048 solo responses per arm, Tandem's marginal token distribution stays
closer to the base on several diagnostics:

![Supplemental summary of documented Qwen3-4B macro accuracy and marginal token drift.](figures/qwen3-4b-summary.png)

Supplemental summary bars; [values and plotting script](figures/README.md#reproducible-summary-plots).

| Diagnostic | Reported result |
|---|---|
| Jensen–Shannon divergence from base | GRPO 0.00355272; Tandem 0.00301511 nats |
| Reduction in divergence | **15.13%**, versus the paper's 37% |
| Selected top-500 shifted tokens ending closer to base | **83.2%** |
| Opposite-direction movement / Spearman correlation | 84.0% / −0.548 |

This partially supports reduced drift. The drift-survival curves cross, and the
selected examples do not show a simple markup-versus-natural-language split.
These are pooled token-frequency statistics with documented reconstruction
choices and no bootstrap intervals; they do not measure human readability or
the junior's conditional cross-entropy. See [definitions, choices, and results](FIGURE3.md).

## Asymmetric benchmark figure: Qwen3.5-9B / 2B

The first main extension trains an initially stronger 9B senior with a frozen 2B
junior. Both step-200 policies are released, and the complete comparison covers
all 1,064 benchmark problems, with 32 solo samples and eight handoff chains per problem.

![Complete Figure 2 evaluation of Qwen3.5-9B seniors with a frozen Qwen3.5-2B junior.](figures/figure2-qwen35-9b-2b.png)

Top: paragraph handoff with the 2B junior. Bottom: solo, including both initial
models. Bands: ±1 bootstrap standard error, from 2,000 problem-bootstrap replicates.

The plotted results show higher macro handoff scores for Tandem and broadly
similar solo scores. Handoff gains appear on AMC, AIME and Olympiad, while
Minerva is slightly lower. [Protocol and completed comparison](QWEN35_PAIR.md) ·
[Figure provenance](figures/README.md#benchmark-exports).

## Pilot results: accuracy versus length pressure

The second main extension asks whether correctness-only solo RL develops useful
shorthand, and whether a token cost strengthens that effect.

![Matrix and RuleTaker accuracy for the base, correctness-only RL and length-penalized RL.](figures/pilot-summary.png)

Correctness-only RL improves these tasks; adding the tested length cost lowers
matrix accuracy from 92.38% to 15.63% and RuleTaker accuracy from 79.49% to 62.11%.
The chart uses the 3K-trained matrix arms at 3K evaluation and the 4K RuleTaker
arms at 4K evaluation; RuleTaker requires all four answers correct. These are
final-step solo-RL pilots, and partial blinded review did not identify opaque shorthand.
[Controls and diagnosis](LENGTH_PENALTY_PILOTS.md) · [All variation workflows](../experiments/README.md).

## Other studies and work in progress

| Study | Type | Recorded status |
| --- | --- | --- |
| [APPS coding](APPS_RUN.md) | Domain-transfer experiment | Both step-200 policies released; sharded final evaluation and grader recovery recorded, with no final aggregate comparison. |
| [Method Atlas](METHOD_AUDIT.md) | Dataset and mathematical audit | Released 374 answer-checked problems with 1,840 labeled solution paths. |
| [Elegance RL](ELEGANCE_RL.md) | Reward/generalization experiment | Training and recovery checkpoints recorded; final comparison pending. |
| [MedCalc](../experiments/medcalc/README.md) | Implemented pipeline | Preparation, training and evaluation code; no empirical results reported. |
| [Natural-language drift](NATURAL_DRIFT_PILOT.md) | Exploratory pilot | Qwen submissions and gated-model access limits recorded; no final results reported. |
| [Native word handoff and recovery](#word-handoff-and-error-recovery) | Interim evaluation and case study | Partial matched comparison and error attribution; controlled recovery experiment remains proposed. |

## Word handoff and error recovery

The [native-word comparison](REPRODUCTION_AUDIT.md#reconnected-scheduling-audit--2026-10-09-late-pacific)
reports an **interim 737-problem** macro pass@8 of 58.44% for GRPO and 58.72% for
Tandem. No final matched comparison is recorded. The latest math-run update is
October 10: GRPO completed; Tandem had 993/1,064 saved problems before replacement.
This is a historical status, not a live queue check. Word evaluation also changes
decoding to the training settings, so it does not isolate switching schedule.

The [manual case study](WORD_HANDOFF_CASE_STUDY.md) finds mistakes and repairs by
both models, with no demonstrated Tandem recovery advantage. On one selected
arithmetic problem, nine failed traces had four base-authored and five
senior-authored first substantive errors. Across the 737 matched problems,
62.6% of GRPO failures and 63.6% of Tandem failures ended at the token limit.
The proposed matched-prefix recovery experiment remains a proposal.

## What is available to verify

This checkout includes both full Figure 2 PNGs, the evaluation code, benchmark
data, experiment records and [training-summary artifacts](../results/training-progress/completed-20261008.json).
The [plot index](figures/README.md) records export provenance, source availability
and reproducible summary data. Follow the [current workflows](WORKFLOWS.md) to run
the pipeline or the [trace-browser guide](COMPARISON_BROWSER.md) to inspect traces.
