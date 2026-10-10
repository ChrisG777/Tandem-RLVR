# APPS coding experiment

This single-seed experiment extends Figure 2 to APPS introductory/interview tasks.

## Interfaces

| Entry point | Contract |
|---|---|
| [`build_apps`](../experiments/coding/build_apps.py) | Immutable dataset revision → train, 200 validation tasks, tiered test Parquets and audit manifest |
| [`compute_score`](../experiments/coding/reward.py) | Response plus normalized tests → binary reward and numeric failure diagnostics; infrastructure failures raise |
| [`campaign`](../experiments/coding/campaign.py) | Sandbox preflight, evaluator dispatch, calibration gate, checkpoint selection, panel manifest |
| [`submit`](../experiments/coding/submit.py) | Explicit site resources → one dependency-gated Slurm campaign with durable job IDs |
| [`setup_grader`](../experiments/coding/setup_grader.py) | Pinned verl checker and uv runtime → isolated CPU grader directory |

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
4096/3000/8192 tokens. The last complete Python block is graded without repair.

The checker is from the pinned verl commit. Its obsolete `pyext` dynamic-module
loader is replaced with `types.ModuleType` and standard `compile`/`exec`; test and
comparison semantics are preserved. Call-based arguments are serialized individually
as JSON lines for this checker. The runtime records the original and adapted hashes.

Every candidate gets a fresh Bubblewrap user/PID/mount/network namespace, a read-only
runtime and input, private `/tmp`, minimal `/dev`, and no project/home/GPU mounts.
System `/usr` supplies dynamic libraries. Limits: 2 GiB address space, 1 MiB output
files, no subprocess creation, five seconds per test, and an outer watchdog of
`5*(test_count+1)+10` seconds. Startup/checker failures raise; they are not wrong answers.

Calibration samples eight solo and eight native-word completions for every validation
task at 0.6/1/-1 decoding. Both modes must have 5–90% accuracy, ≥10% mixed-success
groups, and ≤5% truncation. Failure stops the dependency chain before training.

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

Use `uv sync --project experiments/coding/env --locked` for CPU tools. Unit tests:

```bash
uv run --project experiments/coding/env --with-requirements env/grading-requirements.txt \
  python -m unittest tests.test_coding tests.test_eval_resume tests.test_figure2 \
  tests.test_training_checkpoint tests.test_word_handoff tests.test_math_grading
```

On a cluster, set `REPO`, `RUN_ROOT`, `DATA_ROOT`, `DATA_REVISION`, `UV_BIN_DIR`,
`TANDEM_ENV`, caches, and the existing container/runtime variables. `RUN_ROOT/base.json`
must identify the exact base revision used by Figure 2. `TANDEM_ENV` is an existing,
verified pinned patched training environment; setup never rewrites it.

```bash
uv run --project experiments/coding/env python -m experiments.coding.submit \
  --cpu-args "$CPU_SBATCH_ARGS" --eval-args "$EVAL_SBATCH_ARGS" \
  --train-args "$TRAIN_SBATCH_ARGS"
```

Supply explicit account/QoS/partitions, CPUs, host RAM and walltime after applying
the shared allocation policy. Evaluation args include one GPU; training GPU counts
are added per arm. Existing trainer placement uses one senior card and a separate
junior card for Tandem. Do not copy historical resource defaults without checking fit.

`jobs.json` is created exclusively to prevent duplicate campaigns. Partial submissions
remain recorded and require reconciliation with Slurm before further submissions.
Requeues restore exact compatible generation progress or the latest optimizer state.
Never change a checkout or grader runtime used by active jobs.

Outputs: `data/manifest.json`, `calibration/report.json`, per-arm `selected.json`,
`models.json`, `results/<arm>/<phase>.json`, and `results/apps-figure2.{json,png,svg}`.
