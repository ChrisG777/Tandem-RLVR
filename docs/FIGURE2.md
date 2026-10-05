# Figure 2 reproduction

This evaluates the authors' released base, GRPO, and Tandem checkpoints; it does
not reproduce training. The immutable model revisions are in
[`eval/figure2-models.json`](../eval/figure2-models.json). The trained releases have
no model cards or step metadata, so their correspondence to the paper's selected
GRPO step 200 and Tandem step 120 cannot be independently confirmed.

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
| [`download_figure2.py`](../eval/download_figure2.py) | model manifest → cached weights and `--out` snapshot-path JSON |
| [`figure2.sbatch`](../slurm/figure2.sbatch) | cached models, two GPUs → three solo and two handoff JSONs, figure |
| [`figure2.py`](../eval/figure2.py) | `--results` directory → `--out` stem with `.png`, `.svg`, `.json` |
| [`summarize`](../eval/figure2.py) | full evaluation JSON → per-benchmark and macro pass@k means/SEs in percent |

## Running

All commands run from the repository root. Use uv for Python and dependencies.
Set `REPO`, `TANDEM_ENV`, `HF_HOME`, and `UV_CACHE_DIR` for the cluster. Create
`logs/`, then run `bash slurm/figure2-setup.sh` on a CPU allocation. This installs
the evaluation pins, records the resolved environment, and downloads immutable
model snapshots to `HF_HOME`.

Submit `slurm/figure2.sbatch` with the site's account, partition and QoS options.
It requests two GPUs, 16 CPUs, 128 GB host RAM, and six hours. For Engaging use
`--partition=mit_normal_gpu --account=mit_general`; for CSAIL use a compatible
`vision-shared-*` partition list with `--account=vision-torralba
--qos=shared-if-available`. The script includes `--requeue`.

For a smoke test use `LIMIT=2 SOLO_N=2 HANDOFF_N=2
RESULTS_ROOT=results/figure2-smoke` and a short walltime. Default execution uses
`results/figure2`. Do not mix smoke or changed-protocol outputs with full outputs:
existing completed phase JSONs are skipped on restart. Full phases are atomic;
an interrupted phase must be rerun. Plotting rejects incomplete benchmark panels,
insufficient sample counts, wrong decoding, and handoff round-cap truncation.

Regenerate the plot without GPUs:

```bash
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline eval/figure2.py \
  --results results/figure2 --out results/figure2/figure2
```

Raw generations remain in `results/figure2/{base,grpo,tandem}/`. Preserve them on
the cluster; publish only compact scores, provenance, and the plot through git.
The existing evaluator is unchanged except that its shared k list now extends to
32 (values greater than the number of sampled completions are still omitted).
