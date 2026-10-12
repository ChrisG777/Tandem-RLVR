# APPS pilot run: 2026-10-10

## Concurrency qualification: 2026-10-11

Interactive allocation **2624911** on one H200 (`isola-h200-1`, six CPUs,
148 GiB host RAM) reuses the pinned runtime and dataset. Profile artifacts live
under `/data/vision/torralba/u/chrisge/tandem-rlvr/apps-concurrency-20261011`.
Code `a9acab0` adds bounded `MODE=profile` runs and explicit sequence/cache controls.
The 33 focused coding, sandbox, senior-gate, and word-state tests passed on-node.

Eight-sequence GRPO with a 20-GiB cache completed two steps: generation took
320.85/358.84 seconds and complete steps took 466.99/506.89 seconds. The previous
two-sequence configuration took 1,027.49/1,503.93 seconds for generation and
1,165.43/1,661.23 seconds per step on the corresponding seed-42 batches.
Samples change with batching, so compare response lengths as well as walltime;
these are execution measurements, not a model-quality comparison.

At 16 sequences/32 GiB per model, GRPO steps took 356.28, 444.85, and 505.05
seconds. Its validation exposed a candidate recursion segfault being misclassified
as infrastructure failure; the old profile was interrupted and is not a successful
validation qualification. Fix `bc0941c` distinguishes candidate-start from checker
readiness and classifies candidate crashes/silent exits as zero-reward runtime
failures. An isolated `grader-v2` preserves the old runtime and calibration identity.
The full regression suite, including stale-grader admission checks, passed 69 tests.

The corrected 16-sequence Tandem screening step took 482.46 seconds (344.84 seconds
generation), with senior-token fraction 0.498 and finite gradients. It was stopped
after that completed step to test 32 sequences/48 GiB per model in `c32`, with three
steps and full validation for each arm. The cache calculation uses the actual
4,096-prompt + 6,000-response cap, not the unused remainder of the 12,288-token
model context. Full training remains stopped pending calibration admission and
a complete-run forecast below 48 GPU-hours. Existing calibration must be regraded
before admission with the revised grader; do not silently rewrite its identity.
Other inspected RL configurations already use sequence limits of 1,024
(Qwen3.5/elegance) or 32 (NLA); they do not share APPS's two-sequence restriction.

## Recovery: 2026-10-11 evening

Budget update: do not launch a run projected to exceed **48 GPU-hours**.
Forecast the complete run, including validation, checkpointing, startup, and
GPU count, from measured representative steps before submission. A 24-hour
resumable allocation does not satisfy this total-compute limit by itself.
APPS training is stopped following the completed calibration gate failure
(native truncation 161/1,600 versus the 160/1,600 limit). Independently, the
observed training-only projections exceed the new budget: about 89 GPU-hours
for GRPO and 159 for Tandem at 200 steps using the mean observed step times.
Do not restart this configuration merely by relaxing the truncation gate.

Replacement full-training jobs were **2624591 (GRPO)** and **2624592 (Tandem)**;
both were cancelled after the completed calibration report failed its gate.
Native calibration **2624593** completed all 200 problems. The gate was evaluated
on the existing CPU allocation, and redundant queued gate job **2624594** was
cancelled. All three GPU jobs started on H200s with 24-hour limits. Training uses the
existing 6,000-response/12,288-context protocol and now saves recovery state
every five steps; validation and retained HF exports remain every 20 steps.
Both arms restart from the pinned base because the cancelled attempts had not
saved a checkpoint. Their original logs remain intact.

The previous monitor cancelled 2618184/2618185 when interactive calibration
2618086 was preempted. That was incomplete calibration, not a failed gate.
The replacement gate runs only after calibration completes successfully and
cancels these two training IDs only when the report explicitly says `passed=false`.
Missing reports and interrupted allocations do not cancel provisional training.
Calibration retains native concurrency four to match its saved evaluation identity.

The training checkout remains on main; its wrapper includes the two local changes
for configurable recovery save frequency and calibration concurrency. Runtime
sandbox preflight passed in both replacement training allocations. Submission IDs
are recorded remotely in `recovery-jobs-20261011.txt`; the gate script is
`recovery-gate.sh`. Historical status and monitor descriptions below are superseded
by this section.

Status (2026-10-11): the user authorized full training while calibration continues.
The original held jobs 2618107/2618108 were cancelled because their submitted
scripts enforced the previous gate. Replacement full runs are **2618184 (GRPO)**
and **2618185 (Tandem)**, submitted without calibration or smoke dependencies.
Both retain full-weight training, one compatible GPU, and 24-hour resumable chunks.
Native calibration and independent checks continue in interactive job 2618086.

## Code and allocation

Changes are on `main`; unrelated working-tree edits are preserved. The isolated
cluster checkout is `/data/scratch/chrisge/Tandem-RLVR-apps-20261010`, also on main.
The earlier Engaging CPU setup submission was cancelled when interactive testing
was requested. All subsequent setup, tests and GPU checks reuse CSAIL interactive
allocation **2615728**, `isola-h100-1`: one H100 80 GB, six CPUs, 148 GiB host RAM,
24-hour limit. No duplicate calibration or training submissions exist.

The request admitted compatible H100/H200/A100-80GB hardware through the valid
vision-shared route. Host RAM was budgeted from prior training/offload measurements,
not from the smaller inference checks. GPU training placement remains subject to
the required three-step smoke tests after calibration passes.

Interactive allocations do not automatically requeue. Saved evaluation progress
is resumable, and production batch wrappers implement bounded same-job requeue.
Do not confuse these two operational guarantees.

## Artifacts

Remote run root:
`/data/vision/torralba/u/chrisge/tandem-rlvr/apps-20261010`.
Completed manifests and checks are also copied to local
`outputs/apps-20261010/` (ignored by Git).

| Artifact | Meaning |
|---|---|
| `data/manifest.json` | Immutable APPS revision, split IDs, hashes and exclusion audit |
| `protocol.json` | Pinned base, decoding, budgets, grader identity and gate/training/evaluation settings |
| `final-node-tests.log`, `math-regression.log` | 63 focused tests plus four math-grading regression tests passed on the allocated node |
| `checks/reference-solutions.json` | Four original APPS reference solutions passed: two stdin and two call-based tasks |
| `checks/native.json`, `checks/paragraph.json` | Actual GPU inference and executable reward smoke traces |
| `checks/response-ce.json` | Junior full-response cross-entropy smoke: 16 saved responses scored |
| `checks/regrade-receipt.json` | Sandbox hardening regraded 16 saved tasks, preserving text and changing no scores |
| `calibration/solo.json.progress.json` | Atomic solo calibration progress; partial until the final JSON exists |
| `calibration/word-handoff.json.progress.json` | Native-word progress, committed after each problem |
| `calibration/report.json` | Gate report, created only after both complete 200-task/eight-sample runs |
| `calibration/solo-report.json` | Completed solo-mode evidence; explicitly marks native calibration as pending |

The source dataset revision is `21e74ddf8de1a21436da12e3e653065c5213e9d1`.
Preparation retained 3,890 training problems, 200 validation problems, and all
4,000 introductory/interview test problems (1,000 introductory; 3,000 interview).
The audit excluded 549 training problems with empty/malformed tests. No retained
test problems exceed the prompt limit. A valid large-integer test was preserved
by disabling Python 3.11's integer-string digit cap inside the bounded runtime.

Solo calibration was deliberately interrupted after a completed batch to test
resumption. The saved texts are retained; the final restart receipt verifies that
the completed run preserves all 16 saved problem records. Native-word calibration
and the gate follow in the same allocation. Native inference uses eight active
sequences with correspondingly sized caches on this already-allocated H100; the
smaller default remains available for smaller GPUs. A failed gate stops training; passing
the gate admits independent three-step GRPO/Tandem smoke runs before full training.

## Revised-gate checks and conditional full runs (2026-10-11)

The 10% boundary checks and existing regressions passed on allocation 2618086:
66 focused tests plus four math-grading tests. `checks/pretraining-audit.json`
verifies all 200 validation task IDs and all 1,600 reward/token records. Of 136
truncated samples, 49 have complete Python blocks, 45 compile, and 10 pass tests.
`checks/reward-reproducibility.json` confirms identical diagnostics when regrading
14 saved samples across tiers, pass/fail outcomes and truncation status.
`checks/word-restart.json` confirms all 112 saved native records are unchanged.

Full-weight runs **2618107 (GRPO)** and **2618108 (Tandem)** are submitted on hold,
with one GPU, six CPUs, 148 GiB host RAM and a 24-hour resumable chunk each.
Both admit H100/H200/A100-80GB through the vision-shared route. A held job consumes
no GPU. These are the only submitted full runs; do not submit replacements.

Separate one-shot release watchers in login-node tmux `apps-checks` windows
`grpo-release` and `tandem-release` release each arm after the completed calibration
passes and its own three-step smoke writes a verified `selected.json`. They also
require the reward-reproducibility receipt. GRPO need not wait for Tandem's smoke.
If calibration fails or the interactive allocation ends before a smoke passes,
the corresponding held full job is cancelled. Each watcher expires after 24 hours.
`training-jobs.json`, `<arm>-release.log`, and `<arm>-release.json` record this flow.
The interactive allocation exits after its checks and releases; it does not remain
idle indefinitely. Batch runs retain automatic same-job requeue and checkpoint resume.

At submission neither full run had started: native calibration and the training
smokes were still prerequisites. The active checkout is pinned at `be501f7`.

## Provisional training authorization

`provisional-training-jobs.json` is the current submission record; the older
`training-jobs.json` is historical. The new training checkout is an independent
clone on `main` at `/data/scratch/chrisge/Tandem-RLVR-apps-training-20261011`,
commit `7322912`; the active calibration checkout was left unchanged.
The new admission guard and shell syntax were tested on interactive node 2618086.

The former per-arm release watchers were stopped. Login-node tmux `apps-checks`
window `calibration-monitor` now checks the completed calibration report every
30 seconds and cancels both provisional runs if it fails. It also cancels them
if the calibration allocation ends without a report or monitoring exceeds 24 hours.
On success it allows both full runs to continue. Logs and cancellation receipts
are `provisional-training-monitor.{log,json}`. Failed-run artifacts are preserved;
restart requires addressing the calibration failure, not repeatedly submitting
an unchanged failing protocol. The training startup guard independently rejects
an existing failed report, including on automatic requeue.
