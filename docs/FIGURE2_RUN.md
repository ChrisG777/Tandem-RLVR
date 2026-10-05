# Figure 2 attempt: status 2026-10-05 (America/Los_Angeles)

Status: environment verified; GPU smoke test queued for priority, with no start
estimate. **No measured reproduction results yet.**

The run evaluates the released checkpoints, rather than retraining. See
[FIGURE2.md](FIGURE2.md) for revisions, protocol, limitations and reference values.

## Active CSAIL jobs

| Job | Purpose | Dependency |
|---|---|---|
| 2557373 | CPU verification, imports, cached checkpoints, 11 tests | completed successfully |
| 2557458 | Two-GPU smoke test, two problems and two samples per phase | pending: Priority |
| 2557459 | Full Figure 2 evaluation, then PNG/SVG/JSON generation | smoke succeeds |

The GPU jobs request two GPUs each and execute sequentially, each with requeue
enabled. The full run has a six-hour limit. It skips completed phases on retry;
an interrupted phase has to restart. No GPU time has been consumed as of this
status snapshot; all three pinned checkpoints are cached and CPU checks passed.

- Checkout: `/data/scratch/chrisge/Tandem-RLVR`
- Branch: `reproduce-figure2`, pushed to `ChrisG777/Tandem-RLVR`
- Environment: `/data/vision/torralba/u/chrisge/tandem-rlvr/.venv`
- Verification log: `/data/vision/torralba/u/chrisge/tandem-rlvr/setup-2557373.out`
- GPU logs: `logs/figure2-2557458.out`, `logs/figure2-2557459.out` in the checkout
- Model manifest after download: `results/figure2/models.json`
- Full outputs: `results/figure2/{base,grpo,tandem}/{solo,handoff}.json`
  (the base has only solo output)
- Final figure and compact statistics: `results/figure2/figure2.{png,svg,json}`
- Exact resolved dependencies: `logs/figure2-environment.txt`

## Monitoring / continuation

```bash
ssh slurm-login.csail.mit.edu \
  'squeue -j 2557458,2557459; sacct -j 2557373,2557458,2557459 --format=JobID,State,ExitCode,Elapsed'
```

Inspect failures before resubmission. Dependent jobs use
`--kill-on-invalid-dep=yes`, so a failed prerequisite cancels downstream work.
Once complete, inspect the generated figure, compare the measured pass@4 and
AIME handoff gap with the paper, and retrieve compact results through git.
Raw generations remain on the cluster. Do not treat queue submission or the
synthetic local layout check as a successful scientific reproduction.

## Validation and startup findings

- Nine benchmark-aggregation tests and two Figure 2 tests pass locally and on CSAIL.
- CSAIL CPU imports confirm PyTorch 2.10.0+cu128, vLLM 0.19.1, `LLM`, and
  `AutoTokenizer`. GPU execution is still unverified, pending the smoke test.
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
- Use `UV_LINK_MODE=hardlink` with this environment: it shares the Torralba
  filesystem with the package cache. Forced copies were very slow. The first
  interrupted copy left incomplete packages, repaired by reinstalling them.
- The shared cached SymPy 1.14.0 wheel also lacked
  `sympy/parsing/latex/lark/grammar/latex.lark`. A fresh `uv pip install
  --reinstall --no-deps --no-cache sympy==1.14.0` repaired the private environment
  without changing the shared cache. The final verification ran after this fix.
- Earlier failed/cancelled setup and dependent job attempts are superseded by
  the job IDs above. Only this attempt's jobs were managed; unrelated jobs were
  left alone.
