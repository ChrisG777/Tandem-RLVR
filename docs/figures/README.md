# Plots and source files

| Plot | What it shows | Source |
| --- | --- | --- |
| [Qwen3-4B Figure 2](figure2-qwen3-4b.png) | Complete paragraph-handoff and solo pass@k curves across four benchmarks and their macro average | Committed export; [grading audit](../REPRODUCTION_AUDIT.md#completion-update--2026-10-09-pacific) |
| [Qwen3.5 9B/2B Figure 2](figure2-qwen35-9b-2b.png) | Complete asymmetric-pair comparison, including initial 9B and frozen 2B solo baselines | Committed export; [pair protocol and results](../QWEN35_PAIR.md#completed-benchmark-comparison) |
| [Supplemental 4B summary](qwen3-4b-summary.png) / [SVG](qwen3-4b-summary.svg) | Reported macro accuracy and token drift | Numerical summaries in the [grading audit](../REPRODUCTION_AUDIT.md#completion-update--2026-10-09-pacific) and [Figure 3 record](../FIGURE3.md#results-from-the-completed-figure-2-evaluations) |
| [Matrix / RuleTaker pilots](pilot-summary.png) / [SVG](pilot-summary.svg) | Base, correctness-only RL and length-penalized RL accuracy | [Matrix results](../ENGAGING_PILOT.md) and [length-pressure controls](../LENGTH_PENALTY_PILOTS.md#collapse-investigation-2026-10-10) |
| [Completed training curves](../../results/training-progress/completed-20261008.png) / [SVG](../../results/training-progress/completed-20261008.svg) | Qwen3-4B reward, validation, length and truncation | Existing [summary JSON](../../results/training-progress/completed-20261008.json) and [snapshot](../../results/training-progress/completed-20261008.snapshot.json) |
| [Training diagnostics](../../results/training-progress/completed-diagnostics-20261008.png) | Additional recorded training health checks | Existing training artifact; these are monitoring curves |
| [Earlier solo-only snapshot](../../results/training-progress/solo-20261008.png) / [SVG](../../results/training-progress/solo-20261008.svg) | GRPO training before the two-arm plot was assembled | Historical snapshot; use the completed comparison for the main report |
| [APPS Figure 2 (provisional)](apps-figure2-provisional.png) / [SVG](apps-figure2-provisional.svg) | Step-200 GRPO/Tandem paragraph-handoff (k≤8) and solo (k≤32) pass@k on APPS introductory (1,000) and interview (3,000) plus their macro average; base solo curve pending; 2/32,000 round-capped handoff chains per arm graded as-is | [Scores](apps-figure2-provisional.json); [APPS run record](../APPS_RUN.md) |
| [APPS training curves](apps-training-progress.png) / [SVG](apps-training-progress.svg) | Step-200 runs' training reward, own-mode validation pass@1, response length and truncation; GRPO step-160 metrics were not logged | [Summary](apps-training-progress.json); [solo diagnostics](apps-solo-diagnostics.json) |
| [Original paper overview](../teaser.png) | Paper method and headline claims | Inherited paper illustration, not this repository's replication scores |

PNG/SVG pairs are alternate formats of the same figure. Benchmark x-axes show
the number of attempts, `k`. Training-curve x-axes show optimizer steps; faint
lines are individual steps and bold lines are trailing ten-step means.

<a id="pending-9b2b-benchmark-figure"></a>

## Benchmark exports

Both full Figure 2 PNGs were added upstream in commit `020be73` and are embedded
directly in the [README](../../README.md) and [results guide](../RESULTS.md).
Each covers all 1,064 problems: AMC 121, AIME 90, Minerva 272 and Olympiad 581.
The top row shows paragraph handoff and the bottom row solo performance;
bands are ±1 problem-bootstrap standard error from 2,000 replicates.
Solo uses 32 samples/problem and handoff eight chains/problem.

| Export | Checkpoints | SHA-256 |
| --- | --- | --- |
| `figure2-qwen3-4b.png` | GRPO 160; Tandem 180; frozen Qwen3-4B base junior | `afe84713fbab5c89e14b5efeff3f6368b8d91a30ff7d21c0f745c0f2d543bbe8` |
| `figure2-qwen35-9b-2b.png` | Both seniors at step 200; frozen Qwen3.5-2B junior | `2f41fbd1ee79905d88662566bcb57b138d6e0afbff312a7a8f7cbc8d07683071` |

The exports were supplied without their score JSONs or per-response generation
files. Exact 4B summary values are documented in the grading audit; the 9B/2B
discussion describes the plotted pattern without extracting approximate numbers.
Figure 3 token counts and the full four-panel drift export remain local/cluster
artifacts. The generators are [Figure 2](../../eval/figure2.py) and
[Figure 3](../../eval/figure3.py). See the [asymmetric results section](../RESULTS.md#asymmetric-benchmark-figure-qwen35-9b--2b).

## Reproducible summary plots

[`reported-metrics.json`](reported-metrics.json) transcribes the documented values
and their sources. [`plot_reported_metrics.py`](plot_reported_metrics.py) renders
the two summary figures; it does not interpolate pass@k curves or invent error bars.
The plots use serif fonts and LaTeX text. To regenerate them with matplotlib,
LaTeX and `dvipng` installed:

```bash
python3 docs/figures/plot_reported_metrics.py
```

These bars supplement the full Figure 2 curves. The pilot chart compares the
3K-trained matrix policies at 3K evaluation and the 4K-trained RuleTaker policies
at 4K evaluation, using final step-100 weights. It does not mix in the separate
2K-trained matrix arm.
