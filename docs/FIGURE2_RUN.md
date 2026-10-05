# Figure 2 attempt: status 2026-10-05 (America/Los_Angeles)

**Superseded: all released-checkpoint jobs below are cancelled.** The active
attempt trains both arms from the official Qwen base; see
[FRESH_TRAINING.md](FRESH_TRAINING.md). No author-trained weights enter the new run.

Fresh training campaign: `/data/vision/torralba/u/chrisge/tandem-rlvr/fresh-20261005`.
Separate training uv environment: `/data/vision/torralba/u/chrisge/tandem-rlvr/train-venv`.
Setup job 2559076 is running. Queued jobs:

| Arm | Three-step smoke | 200-step full | CPU cancellation monitor |
|---|---|---|---|
| GRPO | 2559081 | 2559082 | 2559083 |
| Tandem | 2559084 | 2559085 | 2559086 |

All GPU jobs require setup completion. Full jobs additionally require monitor
startup, with no dependency on smoke success. Training has not yet been verified
on GPU. See `logs/train-setup-2559076.out` and `logs/{grpo,tandem}-{smoke,full,watch}-JOB.out`.

The complete downstream pipeline is also queued: checkpoint selection 2559145
waits for the two full and two smoke training jobs; evaluation smoke 2559146 and
full 2559147 wait for our selected checkpoint manifest. Evaluation guard 2559148
starts after selection and cancels 2559147 if evaluation smoke fails. Full
evaluation requires guard startup, not evaluation smoke success. Its 24-hour
allocation writes `eval-full/figure2.{png,svg,json}` under the campaign root.
Invalid training dependencies cancel downstream jobs automatically. No measured
results exist yet.

## Historical released-checkpoint attempt (cancelled)

Status: environment verified; corrected smoke and full GPU jobs are queued
independently, with a running CPU failure monitor. **No measured reproduction
results yet.** Full results remain provisional until smoke artifact checks pass.

The run evaluates the released checkpoints, rather than retraining. See
[FIGURE2.md](FIGURE2.md) for revisions, protocol, limitations and reference values.

## Active CSAIL jobs

| Job | Purpose | Dependency |
|---|---|---|
| 2557373 | CPU verification, imports, cached checkpoints, 11 tests | completed successfully |
| 2559028 | Corrected two-GPU smoke test, two problems and two samples per phase | none |
| 2559029 | Speculative full Figure 2 evaluation, then PNG/SVG/JSON generation | none; eligible independently |
| 2559040 | CPU smoke-state and artifact monitor | running; cancels 2559029 on failure |

The GPU jobs request two GPUs each and may execute concurrently, each with requeue
enabled. The full run has a six-hour limit. It skips completed phases on retry;
an interrupted phase has to restart. All three pinned checkpoints are cached and
CPU checks passed. The earlier smoke job 2557458 failed after 56 seconds because
FlashInfer attempted to write into an inaccessible AFS cache; its dependent full
job 2557459 was cancelled. The new launcher sets FLASHINFER_WORKSPACE_BASE to
node-local scratch. Earlier artifacts are preserved in separate directories.

- Checkout: `/data/scratch/chrisge/Tandem-RLVR`
- Branch: `reproduce-figure2`, pushed to `ChrisG777/Tandem-RLVR`
- Environment: `/data/vision/torralba/u/chrisge/tandem-rlvr/.venv`
- Verification log: `/data/vision/torralba/u/chrisge/tandem-rlvr/setup-2557373.out`
- GPU logs: `logs/figure2-2559028.out`, `logs/figure2-2559029.out` in the checkout
- Monitor log: `logs/watch-2559040.out`
- Monitor status: `results/figure2-flashinfer/smoke-status.json`
- Model manifest after download: `results/figure2/models.json`
- Full outputs: `results/figure2-flashinfer/{base,grpo,tandem}/{solo,handoff}.json`
  (the base has only solo output)
- Final figure and compact statistics: `results/figure2-flashinfer/figure2.{png,svg,json}`
- Exact resolved dependencies: `logs/figure2-environment.txt`

## Monitoring / continuation

```bash
ssh slurm-login.csail.mit.edu \
  'squeue -j 2559028,2559029,2559040; sacct -j 2559028,2559029,2559040 --format=JobID,State,ExitCode,Elapsed'
```

The CPU monitor checks smoke state every 30 seconds and verifies all five smoke
JSONs before writing `passed: true`. Smoke failure, invalid artifacts, repeated
scheduler lookup failures, or monitor termination cancel the full job. The monitor
has a 24-hour limit with a termination signal 60 seconds before expiry. Its launch
sets PATH and UV_CACHE_DIR explicitly for the minimal Slurm environment. No
automatic replacement jobs are submitted. Inspect failures before resubmission.
Once complete, inspect the generated figure, compare the measured pass@4 and
AIME handoff gap with the paper, and retrieve compact results through git.
Raw generations remain on the cluster. Do not treat queue submission or the
synthetic local layout check as a successful scientific reproduction.

## Validation and startup findings

- Nine benchmark-aggregation tests and two Figure 2 tests pass locally and on CSAIL.
- Three local monitor tests confirm cancellation on smoke failure or missing
  artifacts, and preservation of the full job after successful verification.
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
