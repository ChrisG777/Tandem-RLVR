# Figure 2 reproduction

This evaluates independently trained GRPO and Tandem policies, plus their
official Qwen base. Follow [FRESH_TRAINING.md](FRESH_TRAINING.md) first. The base
revision is pinned in [`eval/figure2-models.json`](../eval/figure2-models.json);
`eval/select_figure2.py` builds the evaluation manifest from our own checkpoints.
Authors' RL-trained checkpoints are excluded from the reproduction.

Reference checks from the supplied paper's Table 2 (percentage points):

| Model | Macro solo pass@4 | Macro handoff pass@4 | Handoff / solo |
|---|---:|---:|---:|
| GRPO | 55.56 | 52.40 | 94.3% |
| Tandem | 54.85 | 54.74 | 99.8% |

Section 4.3 additionally reports a 6.6-point Tandem advantage in AIME handoff
pass@8. These reference numbers are comparison targets, never inputs to the plot.

## Protocol

- Use the shipped 1,064 problems: AMC 121, pooled AIME 90, Minerva 272,
  OlympiadBench 581. Macro averages weight these four groups equally.
- Solo: 32 samples/problem, unbiased pass@1,2,4,8,16,32.
- Handoff: 8 chains/problem, pass@1,2,4,8, frozen base junior, senior starts,
  switch on double newlines, shared 3,000-token response budget.
- Existing decoding and grading are unchanged: temperature 0.7, top-p 0.8,
  top-k 20, boxed-answer symbolic grader. Existing deterministic seeds are kept.
- Bands are one bootstrap standard error, resampling problems within each
  benchmark 2,000 times (seed 20261004), with AIME years pooled first.
  Macro SE comes from equal-weight bootstrap benchmark means.
- Figure 2 needs at least 32 solo draws and 8 handoff draws. The paper does not
  specify the total number of draws underlying every plotted estimator; these
  are the minimum sample counts supporting its displayed k values.

Evaluation uses the pinned upstream vLLM 0.19.1 wheel. The repository's patched
vLLM and verl implement word-level tandem **training**, and are not called by
the separate-engine reasoning-step handoff evaluator.

## Entry points

| Entry point | Input → output |
|---|---|
| [`download_figure2.py`](../eval/download_figure2.py) | base-only manifest → cached base and `--out` snapshot-path JSON |
| [`select_figure2.py`](../eval/select_figure2.py) | completed training campaign → base and selected local checkpoint paths |
| [`figure2.sbatch`](../slurm/figure2.sbatch) | cached models, two GPUs → three solo and two handoff JSONs, figure |
| [`figure2.py`](../eval/figure2.py) | `--results` directory → `--out` stem with `.png`, `.svg`, `.json` |
| [`summarize`](../eval/figure2.py) | full evaluation JSON → per-benchmark and macro pass@k means/SEs in percent |
| [`watch_figure2.py`](../slurm/watch_figure2.py) | smoke/full job IDs and smoke outputs → verified status JSON; cancel full job on failure |

## Running

All commands run from the repository root. Use uv for Python and dependencies.
Set `REPO`, `TANDEM_ENV`, `HF_HOME`, and `UV_CACHE_DIR` for the cluster. Create
`logs/`, then run `bash slurm/figure2-setup.sh` on a CPU allocation. This installs
the evaluation pins, records the resolved environment, and downloads immutable
model snapshots to `HF_HOME`.

Submit `slurm/figure2.sbatch` with `MODELS_JSON` pointing at the locally selected
manifest, and the site's account, partition and QoS options.
It requests two GPUs, 16 CPUs, 128 GB host RAM, and six hours. For Engaging use
`--partition=mit_normal_gpu --account=mit_general`; for CSAIL use a compatible
`vision-shared-*` partition list with `--account=vision-torralba
--qos=shared-if-available`. The script includes `--requeue`.
CSAIL batch startup was verified with `--export=NIL`; pass site variables as
`KEY=VALUE` arguments after the script name to avoid inheriting a stale login
environment. Use an absolute `--output` path and `--chdir` to the checkout.

For a smoke test use `LIMIT=2 SOLO_N=2 HANDOFF_N=2
RESULTS_ROOT=results/figure2-smoke` and a short walltime. Default execution uses
`results/figure2`. Do not mix smoke or changed-protocol outputs with full outputs:
existing completed phase JSONs are skipped on restart. Full phases are atomic;
an interrupted phase must be rerun. Plotting rejects incomplete benchmark panels,
insufficient sample counts, wrong decoding, and handoff round-cap truncation.

Queue the full run alongside the smoke job without an `afterok` dependency.
Start `slurm/watch_figure2.py --smoke-job ID --full-job ID --results SMOKE_DIR
--status STATUS_JSON` in an independent CPU allocation. It checks the smoke exit
and all five result artifacts and cancels the full job on failure. Results are
provisional until the status JSON has `passed: true`. A short initial hold on the
full job can protect the interval before the CPU monitor starts; release it as
soon as monitoring is active, without waiting for the smoke job to finish.

Regenerate the plot without GPUs:

```bash
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline eval/figure2.py \
  --results results/figure2 --out results/figure2/figure2
```

Raw generations remain in `results/figure2/{base,grpo,tandem}/`. Preserve them on
the cluster; publish only compact scores, provenance, and the plot through git.
The existing evaluator is unchanged except that its shared k list now extends to
32 (values greater than the number of sampled completions are still omitted).
