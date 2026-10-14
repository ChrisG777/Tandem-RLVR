# Plots and source files

| Plot | What it shows | Source |
| --- | --- | --- |
| Qwen3.5 9B/2B benchmark — **pending** | AMC, AIME, Minerva, Olympiad and macro pass@k; paragraph handoff and solo reasoning | Reserved for original plot and score JSON |
| [Qwen3-4B results](qwen3-4b-summary.png) / [SVG](qwen3-4b-summary.svg) | Corrected macro accuracy and token drift | Numerical summaries in the [grading audit](../REPRODUCTION_AUDIT.md#completion-update--2026-10-09-pacific) and [Figure 3 record](../FIGURE3.md#results-from-the-completed-figure-2-evaluations) |
| [Matrix / RuleTaker pilots](pilot-summary.png) / [SVG](pilot-summary.svg) | Base, correctness-only RL and length-penalized RL accuracy | [Matrix results](../ENGAGING_PILOT.md) and [length-pressure controls](../LENGTH_PENALTY_PILOTS.md#collapse-investigation-2026-10-10) |
| [Completed training curves](../../results/training-progress/completed-20261008.png) / [SVG](../../results/training-progress/completed-20261008.svg) | Qwen3-4B reward, validation, length and truncation | Existing [summary JSON](../../results/training-progress/completed-20261008.json) and [snapshot](../../results/training-progress/completed-20261008.snapshot.json) |
| [Training diagnostics](../../results/training-progress/completed-diagnostics-20261008.png) | Additional recorded training health checks | Existing training artifact; these are monitoring curves |
| [Earlier solo-only snapshot](../../results/training-progress/solo-20261008.png) / [SVG](../../results/training-progress/solo-20261008.svg) | GRPO training before the two-arm plot was assembled | Historical snapshot; use the completed comparison for the main report |
| [Original paper overview](../teaser.png) | Paper method and headline claims | Inherited paper illustration, not this repository's replication scores |

The original checkout contains four distinct figures: the paper overview and
three training figures. Their PNG/SVG pairs are alternate formats, not additional
experiments. In the training plots, orange is GRPO and purple is Tandem; the
solo-only snapshot has only orange. The faint lines are individual steps and the
bold lines are trailing ten-step means. These x-axes show training steps, while
the missing final benchmark curves would show the number of attempts, `k`.

## Pending 9B/2B benchmark figure

A reference screenshot was supplied during review. The original plot export and
underlying score JSON are not available in the checkout, so the README keeps an
explicit slot instead of inferring exact scores from pixels. To fill it, add:

- The original PNG/SVG and score JSON, with their source and hashes.
- The model revisions, selected checkpoints and evaluation settings.
- Completed benchmark/sample coverage and the definition of the uncertainty bands.

Then update the [README slot](../../README.md#asymmetric-benchmark-9b-senior--2b-junior)
and [results section](../RESULTS.md#asymmetric-benchmark-figure-qwen35-9b--2b).

## Reproducible summary plots

[`reported-metrics.json`](reported-metrics.json) transcribes the documented values
and their sources. [`plot_reported_metrics.py`](plot_reported_metrics.py) renders
the two summary figures; it does not interpolate pass@k curves or invent error bars.
The plots use serif fonts and LaTeX text. To regenerate them with matplotlib,
LaTeX and `dvipng` installed:

```bash
python docs/figures/plot_reported_metrics.py
```

The full original 4B benchmark figure and token-drift panels are referenced in
the records as `results/figure2-regraded/figure2-complete.png` and
`results/figure3/figure3.png`, but those files and their underlying evaluation
artifacts are not committed. Their generators are [Figure 2](../../eval/figure2.py)
and [Figure 3](../../eval/figure3.py).
