# APPS coding experiment

This single-seed experiment extends Figure 2 to APPS introductory/interview tasks.
See the [interactive run record](APPS_RUN.md) for current evidence and pending stages.

## Interfaces

| Entry point | Contract |
|---|---|
| [`build_apps`](../experiments/coding/build_apps.py) | Immutable dataset revision → train, 200 validation tasks, tiered test Parquets and audit manifest |
| [`compute_score`](../experiments/coding/reward.py) | Response plus normalized tests → binary reward and numeric failure diagnostics; infrastructure failures raise |
| [`campaign`](../experiments/coding/campaign.py) | Sandbox preflight, evaluator dispatch, manifest-checked calibration gate, checkpoint selection, protocol and panel manifests |
| [`submit`](../experiments/coding/submit.py) | Explicit site resources → one dependency-gated Slurm campaign with durable job IDs |
| [`interactive.sh`](../experiments/coding/interactive.sh) | Existing allocation and prepared inputs → regression checks, calibration gate and independent training smokes; submits no jobs |
| [`setup_grader`](../experiments/coding/setup_grader.py) | Pinned verl checker and uv runtime → isolated CPU grader directory |
| [`regrade`](../experiments/coding/regrade.py) | Saved validation responses → resumable isolated regrading with original scores and texts preserved |
| [`validation_pass4`](../experiments/coding/validation.py) | Manifest-matched four-response validation groups → exact pooled pass@4 for checkpoint selection; incomplete or foreign tasks raise |

## Protocol

The dataset revision is supplied explicitly. The source JSONL is loaded using
the maintained datasets JSON loader, not its legacy executable dataset script.
Normalize statement whitespace for duplicate detection; remove training matches
to any official test statement and duplicate training statements. Exclude malformed
tests and overlength training examples before drawing 100 validation tasks per tier
with seed 0. Test overlength examples stay in the denominator with zero success.
The manifest records every retained task ID, file hash and exclusion.

Prompts contain statements, required I/O conventions, starter code, and an instruction
to reason step by step then provide a complete fenced Python solution. Reference
solutions and grader tests are never inserted. Prompt/response/context limits are
4096/6000/12288 tokens. The last complete Python block is graded without repair.

The checker is from the pinned verl commit. Its obsolete `pyext` dynamic-module
loader is replaced with `types.ModuleType` and standard `compile`/`exec`; test and
comparison semantics are preserved. Call-based arguments are serialized individually
as JSON lines for this checker. The runtime records the original and adapted hashes.

Every candidate gets a fresh Bubblewrap user/PID/mount/network namespace, a read-only
runtime and input, private read-only `/tmp`, minimal `/dev`, and no project/home/GPU mounts.
The root and device directories are read-only too, preventing temporary files from
consuming host RAM outside the process memory limit. Stdin is `/dev/null`.
System `/usr` supplies dynamic libraries. Limits: 2 GiB address space, 1 MiB output
files, no subprocess creation, five seconds per test, and an outer watchdog of
`5*(test_count+1)+10` seconds. Startup/checker failures raise; they are not wrong answers.
The worker emits separate readiness and candidate-start markers. A process crash
or silent exit after candidate execution starts is a runtime failure with zero
reward; a crash before that point is an infrastructure failure. Deploy the worker
and reward wrapper together. Training admission checks the calibration report's
grader identity against the current runtime, so a grader change requires regrading
the saved calibration outputs before full training can be admitted.

Calibration samples eight solo and eight native-word completions for every validation
task at 0.6/1/-1 decoding. Both modes must have 5–90% accuracy, ≥10% mixed-success
groups, and ≤10.1% truncation. The user accepted truncation around 10% for the
optimized relaunch; the observed native rate was 161/1,600 (10.0625%). Failure
stops the dependency chain before training.

Three-step independent smoke runs precede 200-step runs from the pinned Figure 2
base. Batch/minibatch/group are 16/8/8, LR 1e-6, clipping .2, data/rollout seed 42,
and no extra KL/entropy/length reward. Tandem uses existing word boundaries, p=.5,
gap cap 32, and zero junior loss. Save/validate every 20 steps; select highest pooled
validation pass@4 in each arm's own rollout mode, breaking ties at the earliest step.

Final solo evaluation uses 32 samples for base/GRPO/Tandem; paragraph and native-word
handoffs use eight for GRPO/Tandem with the frozen base junior. Evaluation decoding
is .7/.8/20. Figure 2 panels show each difficulty tier and their equal-weight macro
average with 2000 problem-bootstrap replicates. Junior CE covers full responses.
Saved outputs retain task IDs, text, execution outcomes and native token authorship.

## Running

Use `uv sync --project experiments/coding/env --locked` for CPU tools. Test in the
same interactive GPU allocation used to validate inference and sandbox isolation;
do not request a new allocation for each test or repair. Use the pinned training
environment for trainer tests and the separate pinned grading environment for
math grading regression tests. The import path avoids unrelated installed `tests`
packages shadowing this repository:

```bash
PYTHONPATH="$REPO/tests:$REPO" uv run --no-project --offline \
  --python "$TANDEM_ENV/bin/python" python -m unittest \
  test_coding test_eval_resume test_figure2 test_training_checkpoint test_word_handoff
PYTHONPATH="$REPO/tests:$REPO" uv run --no-project --offline \
  --python "$GRADING_ENV/bin/python" python -m unittest test_math_grading
```

On a cluster, set `REPO`, `RUN_ROOT`, `DATA_ROOT`, `DATA_REVISION`, `UV_BIN_DIR`,
`TANDEM_ENV`, caches, and the existing container/runtime variables. `RUN_ROOT/base.json`
must identify the exact base revision used by Figure 2. `TANDEM_ENV` is an existing,
verified pinned patched training environment; setup never rewrites it.

After setup, `bash experiments/coding/interactive.sh` runs the regression checks,
both calibration modes, the gate and (only after a passing gate) both three-step
training smokes in the current allocation. Also set `GRADING_ENV` to the separate
math-grading environment. It resumes saved generations and smoke checkpoints.
`NATIVE_CONCURRENCY` defaults to 2; set it only after accounting for the available
GPU memory. Full training resource selection follows the measured smoke results
and a fresh queue check.

Within the existing allocation, run both calibration modes with
`campaign evaluate --phase solo` and `--phase word-handoff`, using `--split validation`,
`--decoding training`, and `--n 8`. Both commands take `--model`, `--junior`,
`--data`, and `--out`. Apply `campaign calibration --data DATA --solo SOLO_JSON
--word WORD_JSON --out REPORT_JSON` after both finish. The gate requires exact
validation IDs, matching dataset/grader identities, and the pinned base snapshot.
Do not run training when that command fails.

Native-word inference optionally accepts `--native-concurrency N` through the
coding campaign (`--concurrency N` on the native evaluator). It scales both KV
caches so all active full contexts fit without eviction, and records this execution
setting in the resume identity. Defaults retain the smaller two-sequence footprint;
four sequences use two 15-GiB caches plus approximately 16 GiB of BF16 weights and
runtime overhead, requiring an appropriately sized single GPU. Use spare memory in
an existing interactive allocation; do not request extra GPUs for this setting.

```bash
uv run --project experiments/coding/env python -m experiments.coding.submit \
  --cpu-args "$CPU_SBATCH_ARGS" --eval-args "$EVAL_SBATCH_ARGS" \
  --train-args "$TRAIN_SBATCH_ARGS"
```

Supply explicit account/QoS/partitions, CPUs, host RAM and walltime after applying
the shared allocation policy. Evaluation args include one GPU; training GPU counts
are added per arm. Coding configures both arms for one GPU, with Tandem's frozen
junior colocated. The optimized configuration uses 32 active sequences and a
bounded 48-GiB KV cache per model on one H200. This does not fit an 80-GiB GPU;
choose compatible hardware explicitly. Both caches cover the maximum permitted
4,096-prompt + 6,000-response tokens without eviction/replay of native sampler
state. The unchanged batch/group sizes queue through those active sequences.
The training smoke must validate fit and authorship before this placement is used
for a full run. Do not copy historical resource defaults without checking fit.

For bounded throughput checks in an existing GPU allocation, use
`MODE=profile PROFILE_STEPS=2 ARM=grpo` (or `tandem`) with `job.sbatch train`.
Use a fresh `RUN_ROOT` containing the pinned `base.json` and a link to the grading
runtime, with `DATA_ROOT` pointing to the prepared dataset. Profiles use full-weight
updates and the unchanged training batch/decoding, but run at most five steps,
save no checkpoints, and do not select a model. They may run independently of a
failed calibration gate; that exception never admits full training.
`ROLLOUT_MAX_SEQS` and `ROLLOUT_KV_BYTES` override the active-sequence limit and
per-model KV cache bytes; defaults are 32 and 48 GiB. Adjust both together
and verify cache capacity, finite losses, executable rewards, and Tandem authorship.
`PROFILE_TEST_FREQ` optionally enables validation to measure its cost. Before any
full launch, forecast all 200 steps plus validation, saves, and startup below
the user's 48-GPU-hour limit. A short allocation or automatic requeue is not a
substitute for this whole-run budget check.
`MODE=smoke SMOKE_VAL_MAX_SAMPLES=200` combines the required three-step
checkpoint-writing smoke with a full validation timing measurement; the default
smoke still validates eight problems.
The qualified GRPO configuration uses 64 sequences and 96 GiB KV on one H200;
Tandem uses the 32-sequence/48-GiB-per-model defaults. See the run record for
measured memory, timing, and complete-run budget evidence.

APPS selection uses `apps/validation/pass@4`, computed from saved validation
responses with the existing pass@k estimator. The pinned verl `best@4` metric
bootstraps with replacement and is retained only as a raw diagnostic. Full runs
require all 200 manifest tasks with exactly four responses at every validation
checkpoint; smoke runs may use eight manifest tasks. Math selection is unchanged.

Under Apptainer, `job.sbatch` prefixes Bubblewrap with
`unshare --user --map-root-user --mount --propagation private`. This makes the
container runtime bindable inside a fresh mount namespace; the parent mounts
remain unchanged. Bubblewrap still creates the isolated candidate sandbox and
startup still requires its isolation/grading preflight. Test this inside the
target container on an allocated compute node when changing deployment settings.

After changing the grader, run `uv run --python "$TANDEM_ENV/bin/python" --no-project
--offline python -m experiments.coding.regrade --input OLD.json --out NEW.json
--data "$DATA_ROOT" --workers 4` for each saved calibration mode. This executes
only saved responses in the sandbox, resumes per problem, preserves original
grades, and rejects changed inputs or grader identity on resume. Then regenerate
the calibration report and protocol with `campaign calibration` and `campaign protocol`.

`jobs.json` is created exclusively to prevent duplicate campaigns. Partial submissions
remain recorded and require reconciliation with Slurm before further submissions.
Requeues restore exact compatible generation progress or the latest optimizer state.
`MAX_RESTARTS` also bounds native preemption retries at job startup; the optimized
relaunch uses three retries and ten-step recovery saves within its budget forecast.
Never change a checkout or grader runtime used by active jobs.

Outputs: `data/manifest.json`, `protocol.json`, `calibration/report.json`, per-arm `selected.json`,
`models.json`, `results/<arm>/<phase>.json`, and `results/apps-figure2.{json,png,svg}`.

The revised coding protocol doubles the response budget to 6,000 tokens and uses
a 12,288-token context after the original solo calibration hit 17.06% truncation.
The 4,096-token prompt limit and validation split are unchanged. Existing dataset
manifests retain their original preparation metadata; `protocol.json` records
the active generation limits. Revised calibration uses a fresh run directory.
Shared evaluators accept `--max-response`; their math default remains 3,000.

On 2026-10-11 the user authorized relaxing the pilot truncation gate from 5%
to 10% after observing 8.5% solo truncation at 6,000 tokens. This is a
validation-informed protocol revision, not a preregistered threshold; no test
scores were consulted. Both modes must still pass all gates, followed by
independent three-step full-weight (LoRA rank zero) training smokes.

The user subsequently authorized full training concurrently with pending
calibration. `campaign admit-training --report PATH --allow-pending` permits
that explicit exception; an existing failed or invalid report always rejects
startup. The training wrapper opts in with `ALLOW_PENDING_CALIBRATION=1`.
Provisional submissions require a cancellation monitor watching the gate report.
