# Figure 2 attempt: status 2026-10-05 (America/Los_Angeles)

## Allocation-policy audit: 2026-10-06 Pacific

Read the synchronized `GPU_ALLOCATION.md` and both cluster guides. Earlier
allocation notes below are historical, not defaults for new submissions.

| Work | Audit and action |
|---|---|
| CSAIL GRPO 2562647 | Running; one 80-GB-class GPU, six CPUs, 144 GiB. Sampled smoke RSS was approximately 144 GiB and observed GPU peak approximately 77 GiB, so shrinking it is unjustified. Existing checkpoint resumption is evidenced by two real restarts. |
| CSAIL Tandem 2560590 | Running; two 80-GB-class GPUs, six effective CPUs, 144 GiB. Current launcher puts the frozen model/cache on a separate device. Two GPUs are a validated placement requirement, not proof that one H200 can never work; colocation would require changing cache sizing and validating the implementation. Preserve this running experiment. Slurm retains stale `TresPerTask=cpu=8` metadata, but actual allocated CPUs and CPUs/task are six; Slurm refused an in-place correction on a running job. |
| CSAIL shorthand 2581354_0–1 | Both completed, in 3m06s/3m09s. No holds or cancellation during this audit. Existing per-budget atomic results provided restart points. |
| CSAIL evaluation 2562652/2562653 | One GPU each, four CPUs, 48 GiB (two concurrent model engines). Changed in place to account `csail`, QoS `shared-if-available`, partitions `csail-shared-h200,csail-shared-l40s` after comparing dry-run estimates with vision-shared. IDs, dependencies and requeue preserved. |
| Engaging base evaluation | Removed H200-only restriction; one untyped GPU on `mit_normal_gpu,mit_preemptable`, QoS `normal`, four CPUs, 32 GiB, six hours, requeue. Live inventory admits A100/A40/L40S/H100/H200/RTX Pro 6000 nodes, excluding smaller GPUs and unavailable/reserved nodes. Inventory and exclusions are recorded in the campaign. |

Torralba H100/H200 GPUs were fully occupied by main/interactive QoS jobs, none
in `vision-torralba-main`'s preemption list. Checked configured CPUs/RAM,
owner pending jobs and reservations as well. Older Torralba hardware does not
meet the existing evaluation memory/runtime envelope. Both shared routes were
dry-run checked; CSAIL-wide estimates were earlier at this snapshot. Estimates
are not reservations and dependency-blocked evaluations need reassessment when
the trained checkpoints become available. Data/runtime locality supports keeping
existing training on CSAIL and the prepared independent base evaluation on Engaging.

Solo/handoff evaluation now atomically checkpoints every 16 problems, with
input-signature checks and successful simulated-interruption tests. The five-phase
legacy CSAIL evaluation remains one serial allocation; splitting those independent
phases would require replacing the already-spooled job and its guard/dependencies.
It was preserved under the instruction not to cancel CSAIL jobs, so this orchestration
aspect is not yet fully aligned with the new policy. No training GPU placement or
optimization settings were changed.

CPU monitor **2582174** is running (one CPU, 256 MiB, two days), monitoring the
two existing training IDs. It can requeue each once before walltime, only with
saved model/optimizer/extra-state/data artifacts and within the launcher's native
three-restart bound. It never retries application failures. Monitor 2582166 failed
at startup because `--export=NIL` omitted PATH; the corrected monitor explicitly
sets scheduler PATH. Mocked continuation/restart tests pass; no actual walltime
continuation has been needed yet.

The broadened Engaging job 25113963 started on an L40S within minutes, then failed
after 47 seconds: Triton's JIT inherited an unmounted host compiler path. The
runtime image also lacked an assembler/toolchain. Replaced the image with official
CUDA 12.8.1 **devel** / Ubuntu 22.04, pinned amd64 digest
`sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7`,
and select container-local GCC/G++. Setup now checks C compilation before admitting
GPU work. Submitted setup **25119986** and dependent evaluation **25119987** on
the broadened route above. These were pending at submission; GPU execution of
the corrected image is not yet verified. Failed outputs are preserved; no GPU
job was manually cancelled. Code and tests were pushed to `main`.

## Priority update: 2026-10-07 00:53 UTC

The critical path is Tandem training (job 2560590, last completed step 49/200),
matched solo GRPO training (2562647, step 71/200), checkpoint verification and
selection (2562651), then solo/handoff evaluation (2562652–2562654). Both
training jobs are running on A100 80 GB GPUs; neither was interrupted or changed.
The dedicated Torralba H200 node had all eight GPUs allocated at this check.
The tables below this update retain historical submission-time states.

Engaging can evaluate the exact pinned base now, and the selected trained
checkpoints later. Submitted CPU environment setup **25113961** and dependent
base-only evaluation **25113963**, both pending at submission. The base job
requests one H200, four CPUs, 48 GiB and six hours, retaining all 1,064 problems,
32 samples/problem and the original decoding/grader. It uses `MODE=base` in the
existing evaluation launcher; implementation `5e19b1a`. GPU execution remains
unverified. Campaign:
`/orcd/scratch/orcd/013/cge7/tandem-rlvr/figure2-engaging-20261007`.

The container setup explicitly uses Engaging's shared Apptainer 1.4.2 executable;
the prior calibration setup failed because the login node's system Apptainer
was absent on its compute node. This setup reuses the pinned evaluation package
installer and base downloader, without launching shorthand calibration.
Completed Engaging results must be verified and transferred before the CSAIL
workflow can reuse them; the existing full evaluator otherwise computes its own
base result. Checkpoints for trained policies are not yet available for final
evaluation and have not been transferred.

Tinker is unsuitable for this exact reproduction: the pinned Qwen3-4B-Instruct-2507
is retired there, and its supported LoRA training differs from these full-parameter
runs. No Tinker work has been launched. The pending CSAIL shorthand array 2581354
was briefly held, then released at the user's request; leave it in the queue.
No CSAIL jobs were cancelled.

**Independent training is in progress; no Figure 2 results yet.** Both arms
initialize from the official Qwen base, with the authors' patched vLLM/verl.
No author-trained weights enter this attempt. See [FRESH_TRAINING.md](FRESH_TRAINING.md).

## Current jobs

| Work | Job | Status at 2026-10-06 00:53 UTC |
|---|---|---|
| Tandem three-step smoke | 2560589 | Completed in 39m12s; checkpoints and metrics verified |
| Tandem 200-step training | 2560590 | Pending; original job and queue position retained |
| Tandem smoke guard | 2560591 | Completed; passed |
| Corrected GRPO three-step smoke | 2562646 | Pending |
| Corrected GRPO 200-step training | 2562647 | Queued independently of smoke success |
| GRPO smoke guard | 2562648 | Running; cancels 2562647 on smoke/artifact failure |
| Checkpoint selection | 2562651 | Waiting for full training and corrected GRPO smoke |
| Evaluation smoke / full | 2562652 / 2562653 | Waiting for selected checkpoints |
| Evaluation guard | 2562654 | Starts after selection; cancels 2562653 on smoke/artifact failure |

Tandem smoke ran on two A100 SXM4 80 GB GPUs. All three optimizer steps had
finite loss and gradients; senior-token fractions were 0.49888, 0.49853 and
0.50249. The saved Hugging Face checkpoint passed artifact checks. Validation
used only eight smoke problems, so it is not a benchmark/reproduction score.
The full run has not started. Its first two smoke steps took 10.4 and 8.1 minutes;
a full 200-step run at that speed will need resumption beyond a 24-hour shared
allocation. There is no reliable queue start estimate. Checkpoints save every
20 steps; resume only the same run's optimizer state if walltime is reached.

Training resources are one GPU for GRPO, two for Tandem, six CPUs and 144 GiB
host RAM each. Completed Tandem smoke peaked at 128.0 GiB RSS, so the original
128 GiB request was increased by 16 GiB. Tandem's pending job was updated in
place. Evaluation remains one GPU, four CPUs, 48 GiB. Training accepts A100
80 GB, H100 and H200; evaluation additionally accepts L40S/A6000/RTX6000Ada.

## GRPO cache recovery

GRPO smoke 2560586 failed before training because vLLM requested 63.34 GiB while
62.45 GiB was free beside the colocated policy. Guard 2560588 correctly cancelled
full run 2560587; invalid dependencies cancelled old selection/evaluation jobs
2560601–2560604. Batch launches now reserve 0.65 of GPU memory for vLLM instead
of vanilla's 0.8, matching the successful Tandem setting. Optimizer, sampling and
global batch settings are unchanged. Launcher argument tests and shell syntax
checks passed; four local verl-method checks were skipped because local verl is
absent. Environment setup 2559076 previously passed all 39 cluster tests.

New campaign: `/data/vision/torralba/u/chrisge/tandem-rlvr/fresh-20261005-cachefix`.
It contains fresh GRPO directories, the same pinned base manifest, and explicit
symlinks to the preserved Tandem smoke/full directories in
`fresh-20261005-rayfix`. No successful training is repeated. Selection rechecks
both arms and initializes the evaluation manifest only from verified runs.
Full jobs require guard startup, **not smoke success**. Evaluation outputs will
be `eval-full/figure2.{png,svg,json}` under the new campaign root.

Remote checkout: `/data/scratch/chrisge/Tandem-RLVR`, branch `reproduce-figure2`.
Training environment: `/data/vision/torralba/u/chrisge/tandem-rlvr/train-venv`.
Evaluation environment: `/data/vision/torralba/u/chrisge/tandem-rlvr/.venv`.
Training logs: `logs/{grpo,tandem}-{smoke,full,watch}-JOB.out`.
Latest job IDs are also recorded in the new campaign's `jobs.txt`.

## Earlier Ray startup recovery

Tandem smoke 2559084 failed after 26 seconds before training because Ray 2.55.1's
`uv run` hook rejected `runtime_env.working_dir=None`. Its full/dependent jobs
were cancelled. `slurm/training-env.sh` disables that hook; Ray uses the existing
shared uv-managed interpreter. CPU job 2560582 verified real worker startup and
both patched module paths. The subsequent Tandem smoke passed on GPU.

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

## Requeue priority over our other CSAIL work (2026-10-07)

At the user's request, reproduction training 2560590/2562647 and its downstream
jobs retain Slurm `Nice=0`. The other active CSAIL jobs, 2582199 (NLA GPU) and
2582214 (NLA CPU monitor), were updated in place from `Nice=0` to `Nice=1000`.
Verified priorities immediately afterward: reproduction training 33, downstream
reproduction 21–26, and both other jobs 1. All running jobs continued, with no
holds, cancellations, or forced restarts.

The shared GPU allocation policy now instructs agents to use `--nice=1000` for
new non-reproduction CSAIL submissions until this reproduction finishes, and
`--nice=0` for the reproduction. Nice remains part of an existing job across
native requeues; newly submitted jobs need the explicit flag. This is queue
preference within Slurm's site rules, not preemption of already running work or
a guarantee of immediate capacity. Engaging priorities are unchanged. Restore
the two changed jobs' original Nice values if they survive the reproduction.

## Engaging checkpoint continuation (2026-10-07)

Both CSAIL training jobs were preempted again at 03:16 / 03:29 PDT, after logging
Solo step 136 and Tandem step 86. Their latest complete checkpoints are steps
120 and 80. The user authorized rsync transfer of those latest checkpoints.
A direct Duo-authenticated CSAIL→Engaging connection avoids the slow laptop relay;
source files are pinned with hard links so later CSAIL checkpoint retention cannot
remove the transferred versions. Optimizer, RNG/scheduler and dataloader state
travel with the model. Scientific settings remain unchanged.

Both imported checkpoints were also their run's best validation checkpoints.
A migration receipt records the source verifier result, verified three-step smoke
run and SHA256 of each original metric log. On Engaging, the verifier checks these
hashes, preserves the source-verified step-20 metric gap, and compares the retained
best checkpoint with every new validation checkpoint. Inferior historical weights
are not transferred. Unsaved steps will be replayed. Original logs are retained,
and new attempts replace abandoned steps through the existing merge logic.

Engaging runs use the pinned patched training environment and working CUDA devel
container, six CPUs/144 GiB per arm, one Solo GPU or two Tandem GPUs (one trainable
rank plus frozen junior). Both normal/preemptible routes remain eligible with
six-hour chunks; save every ten steps, validate every twenty, and resume before
walltime from complete checkpoints. Recovery is bounded at eight restarts for
this migration, allowing the longer Tandem run to span allocations.

A CPU gate waits for successful transfer, verifies datasets/checkpoint history,
and preserves the imported HF weights independently of checkpoint retention.
Only then can GPU training start. CPU checkpoint selection follows both runs;
four independent GPU evaluation phases reuse the completed frozen-base evaluation,
then a CPU job renders Figure 2. CSAIL jobs remain queued during migration.
