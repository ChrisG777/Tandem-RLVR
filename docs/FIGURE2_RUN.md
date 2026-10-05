# Figure 2 attempt: 2026-10-04 (America/Los_Angeles)

Status: submitted; **no measured reproduction results yet**.

The run evaluates the released checkpoints, rather than retraining. See
[FIGURE2.md](FIGURE2.md) for revisions, protocol, limitations and reference values.

## Active CSAIL jobs

| Job | Purpose | Dependency |
|---|---|---|
| 2557058 | CPU environment installation and pinned checkpoint downloads | none |
| 2557064 | Two-GPU smoke test, two problems and two samples per phase | setup succeeds |
| 2557065 | Full Figure 2 evaluation, then PNG/SVG/JSON generation | smoke succeeds |

The GPU jobs request two GPUs each and execute sequentially, each with requeue
enabled. The full run has a six-hour limit. It skips completed phases on retry;
an interrupted phase has to restart. No GPU time has been consumed as of this
status snapshot; the CPU environment install is running.

- Checkout: `/data/scratch/chrisge/Tandem-RLVR`
- Branch: `reproduce-figure2`, pushed to `ChrisG777/Tandem-RLVR`
- Environment: `/data/vision/torralba/u/chrisge/tandem-rlvr/.venv`
- Setup log: `/data/vision/torralba/u/chrisge/tandem-rlvr/setup-2557058.out`
- GPU logs: `logs/figure2-2557064.out`, `logs/figure2-2557065.out` in the checkout
- Model manifest after download: `results/figure2/models.json`
- Full outputs: `results/figure2/{base,grpo,tandem}/{solo,handoff}.json`
  (the base has only solo output)
- Final figure and compact statistics: `results/figure2/figure2.{png,svg,json}`
- Exact resolved dependencies: `logs/figure2-environment.txt`

## Monitoring / continuation

```bash
ssh slurm-login.csail.mit.edu \
  'squeue -j 2557058,2557064,2557065; sacct -j 2557058,2557064,2557065 --format=JobID,State,ExitCode,Elapsed'
```

Inspect failures before resubmission. Dependent jobs use
`--kill-on-invalid-dep=yes`, so a failed prerequisite cancels downstream work.
Once complete, inspect the generated figure, compare the measured pass@4 and
AIME handoff gap with the paper, and retrieve compact results through git.
Raw generations remain on the cluster. Do not treat queue submission or the
synthetic local layout check as a successful scientific reproduction.

## Validation and startup findings

- Nine benchmark-aggregation tests and two Figure 2 tests pass locally.
- Shell syntax and Python compilation pass; the synthetic plot layout was
  visually inspected and is not included as an experimental result.
- The broader training test suite cannot run in the local evaluation-only
  environment: PyTorch and the patched verl/vLLM training modules are absent.
- Engaging's documented `orcd-login001.mit.edu` is deprecated. The replacement
  `orcd-login.mit.edu` required Duo; the authentication attempt timed out.
- CSAIL's expired AFS credentials were renewed using the existing Keychain
  credential without exposing or copying it.
- Batch jobs inheriting the login environment were cancelled at startup; a
  minimal `--export=NIL` diagnostic succeeded. The scripts accept site variables
  as explicit `KEY=VALUE` arguments and tolerate absent HOME/USER variables.
- The shared cached ANTLR 4.9.3 source was missing `bin/pygrun`; setup installs
  that exact version without using or modifying the shared cache entry.
