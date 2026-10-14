# APPS pilot run: 2026-10-10

## Step-200 final evaluation, sharded: 2026-10-13

The user corrected the final-evaluation checkpoints: both arms use
`hf/global_step_200` (verified weights; both runs completed 200 steps), not the
validation-selected GRPO step 60 / Tandem step 40. All earlier final-evaluation
jobs were cancelled and their partial results (175 MB) moved, not deleted, to
`results-selected-step60-step40-20261013/`. `results/final-models-step200.json`
records the decision, checkpoint paths, junior snapshot and code commit `a56b541`.

Diagnosis of the previous jobs on one shared H100 node (4 CPUs each): solo and
paragraph handoff GPUs idled at 0% while every response was graded serially in a
fresh sandbox. Grading one saved 512-response solo batch took 341 s serially and
30 s with 16 concurrent sandboxes, with identical grades; generation of a batch
takes about 210 s. Each batch also rewrote the whole growing progress JSON
(230 MB for solo). Native word handoff is capped at eight active sequences on
80 GB GPUs: the frozen junior holds an equal-sized KV cache, and twelve OOMed.

Interactive H100 measurements (allocation 2653769, problems 2000+): decode CUDA
graphs did not speed up solo (455 s versus 462 s generation for 32 problems at
512 sequences), so solo stays eager; handoff and word use decode graphs. A
16-problem handoff shard took 502 s and an eight-problem word shard 525 s,
both including model loading; word authorship checks passed.

Commit `a56b541` adds `--shard START:END` (contiguous problem ranges with
global-index seeds; resumable `OUT.shard-START-END.progress.json`) and
concurrent APPS grading to solo, paragraph and word evaluators. Rerunning the
unsharded command absorbs complete shards in order; result JSONs list each
shard's file hash and execution settings. Word shards record their concurrency
(8 on 80 GB GPUs, 14 on H200s) and queue four problems per saved batch.

The job wrapper is `apps-shard.sbatch` in the CSAIL run root, submitted with
`--export=NIL`; `--export=ALL` submissions were held by CSAIL user-environment
retrieval failures and resubmitted. At the user's request the work runs as 55
shards (solo ~570–600, handoff ~540–600, word ~360–400 problems each),
alternating Torralba owner priority and vision-shared routes; word shards
require 80 GB-class GPUs. Receipts: `shard-jobs-step200.tsv` (shards) and
`final-jobs-step200.tsv`: one CPU merge per arm/phase after all its shards
succeed, then legibility on GPU after each solo merge. Engaging received a
verified grader/data mirror (`apps-eval-mirror-20261013`), but it lacks the
step-200 checkpoints and its preemptible queue forecast was about a day, so no
Engaging shards were submitted.

Results (2026-10-14): all eight step-200 outputs are complete under
`results/{grpo,tandem}/`; each pass@k result covers all 4,000 test problems after
absorbing its shards. Macro averages over the introductory and interview tiers:

| Step 200 | pass@1 | pass@4 | pass@8 |
|---|---|---|---|
| GRPO solo (n=32) | 0.568 | 0.685 | 0.723 |
| Tandem solo (n=32) | 0.539 | 0.652 | 0.690 |
| GRPO + junior, paragraph handoff | 0.566 | 0.685 | 0.725 |
| Tandem + junior, paragraph handoff | 0.547 | 0.667 | 0.707 |
| GRPO + junior, word handoff | 0.561 | 0.694 | 0.736 |
| Tandem + junior, word handoff | 0.553 | 0.686 | 0.728 |

Solo pass@32 is 0.775 (GRPO) and 0.746 (Tandem). Word-handoff senior token
fractions are 0.500 (GRPO) and 0.502 (Tandem). Junior cross-entropy on solo
responses (128,000 CoTs each) is 0.159 nats for GRPO and 0.147 for Tandem.
Merges ran as 1-GPU jobs on Torralba RTX 3090s after `tig-cpu` forecast starts
over a day away at 1 CPU/12 GB; each took 1-4 minutes. Overnight, shards
preempted onto H200s were rejected on resume because 14 sequences differed from
their recorded 8; the wrapper now reuses a shard's recorded concurrency.

Grader fix (2026-10-14): two shards (Tandem handoff 1000:1600 and solo 2857:3429)
stopped on `TimeoutError: APPS test timeout` reported as an infrastructure error.
The pinned checker returns call-based wrong answers without cancelling its 5 s
alarm, so a candidate that used nearly its whole limit could trigger the alarm
in the worker after the checker returned. Commit `dc66bdd` disarms the alarm when
the checker returns and grades an escaped alarm as a candidate timeout; both
outcomes have zero reward, and recorded rewards are unchanged. The new runtime
`grader-timeout-fix` differs from `grader-output-fix` only in `worker.py` and
passed the sandbox integration tests. `migrate_shard_grader` (`5877caa`) re-keys
old-grader shards after checking the reconstructed old identity, backing up
originals as `*.before-timeout-fix`; it was validated on one completed shard per
phase. The failed and not-yet-started shards were resubmitted on the new grader,
which migrates each job's own range; every merge job migrates its arm/phase first.

## Engaging container repair: 2026-10-12

Jobs 25630667 and 25632334 both failed after three seconds, before model loading
or checkpoint creation. Calibration passed, but Bubblewrap could not bind `/usr`
from Apptainer's unbindable overlay mount. Commit `12a27e1` runs Bubblewrap inside
a new user/mount namespace with private propagation when the container is active.
This changes neither the parent container/host mounts nor grading logic, limits,
or calibration identity. The existing sandbox preflight remains mandatory.

Engaging compute allocation **25639298**, node `node1609`, passed sandbox isolation,
known-correct/failing stdin and call-based grading, and all **72 regression tests**
inside the actual CUDA container. This CPU-only compatibility check needed no GPU;
the earlier H200 throughput qualification is unchanged. The allocation was released.
Receipts are `checks/engaging-sandbox.json` and `engaging-sandbox-regression.log`.

Replacement jobs are **25639349 (Tandem)** and **25639351 (GRPO)**, queued for
priority when checked. They retain the qualified one-H200 configurations, 48-hour
allocations, automatic resume, and 44.82/27.74 GPU-hour forecasts. Failed job IDs
are archived in `training-jobs-before-sandbox-fix.json`; no old continuation is live.

Recent CSAIL history does not establish a reliable six-hour uninterrupted window:
from October 10 through this check, eight H200 attempts were preempted before six
hours, while eight others ran longer than six hours (some 7–14 hours). Short jobs
and early application failures are excluded from this comparison; several attempts
share a preemption event, so it is not an independent survival-rate estimate.
CSAIL-wide shared jobs in this sample fared better than vision-shared jobs.
Torralba's eight H200s remained occupied by a non-preemptible owner job. CSAIL
project storage had only 376 GiB free and scratch was full, versus 754 GiB free
on Engaging. Both new checkpoint histories therefore remain on Engaging.
Engaging's six-hour limit applies to normal GPUs; these jobs use its separate
48-hour preemptible route. Neither cluster guarantees uninterrupted shared access.

## Optimized relaunch: 2026-10-12

The user authorized relaunch after throughput qualification and explicitly accepted
truncation around 10%. The recorded pilot limit is now 10.1%, not a silent override
of the failed 10% report. Regrading all 3,200 saved calibration responses with
`grader-v2` changed zero rewards. Solo/native accuracy is 66.9375%/65.8125%, mixed
groups 28.5%/28%, and truncation 8.5%/10.0625%. All 72 regression tests passed on
the interactive H200; current grader identity passes training admission.

At 32 active sequences and 48 GiB KV per model, Tandem's three complete profile
steps took 393.19, 451.25, and 592.52 seconds; generation took 250.77, 292.88, and
390.96 seconds. Senior fractions were 0.501/0.500/0.502, with finite gradients and
zero aborted responses. Full 200-problem/four-sample validation took 1,457.60
seconds and the profile exited zero. A dataloader cleanup warning occurred after
all metrics were emitted; no grading infrastructure error occurred during validation.

Tandem's checkpoint smoke passed and saved weights in 156.19 seconds. Its validation
trace contains all 32 responses for eight problems. A separate process restored
model, optimizer, RNG, scheduler and dataloader state at step three, then completed
step four in 494.98 seconds. Original smoke records were preserved; the restart
receipt is `checks/tandem-resume.json`.

The final Tandem forecast is **44.82 GPU-hours**, using the slowest observed time
for each of the four measured batches: 26.93 hours for 200 steps, 4.05 for ten
validations, 4.37 for twenty saves at the worst historical save time (787.4 seconds),
and two hours startup/recovery, then 20% headroom. The measured checkpoint cost is
lower than this allowance. Recovery saves are every ten steps; validation and
retained selection exports stay every twenty. Native retry count is bounded at
three. Checkpoint export reached the 148-GiB allocation's cache-reclaim boundary
without an OOM; full jobs request 160 GiB host RAM for additional headroom.

Tandem full job **25630667** was submitted on Engaging with one H200, six CPUs,
160 GiB host RAM, and the route's maximum 48-hour allocation. It starts independently
from the pinned base, uses 32 sequences/48 GiB KV per model, and was queued for
priority at submission. GRPO full job **25632334** uses 64 sequences/96 GiB KV
for its one model, with the same GPU/CPU/RAM/walltime request. Both jobs are
independently queued on Engaging; neither has begun training at this report.

GRPO's completed three-step smoke took 259.97, 264.13, and 315.98 seconds for
training (checkpoint time excluded); generation took 106.62, 106.62, and 118.60
seconds. Full validation took 426.77 seconds, and checkpoint saving took 143.39.
All 800 validation responses matched the 200-task manifest. Its complete-run
forecast is **27.74 GPU-hours**, using the same conservative save allowance,
two-hour startup/recovery allowance and 20% headroom as Tandem. Both budgets
are below the user's 48-GPU-hour cap. These are short-pilot forecasts.

Code `bcb3daf` selects checkpoints by exact `apps/validation/pass@4` from saved
four-response groups, rather than verl's with-replacement bootstrap `best@4`.
Both saved smoke checkpoints passed the new manifest-matched audit. Tandem was
briefly held while this was tested, then released under the same job ID. Its
container entrypoint loads the updated script. The full jobs retain independent
base initialization, recovery saves every ten steps and validation every twenty.
The shared resume path passed a focused process restart. Allocation **2624911**
was released after qualification; no duplicate APPS continuation remains.

The optimized run root is `apps-optimized-20261011` under CSAIL's
`/data/vision/torralba/u/chrisge/tandem-rlvr` and Engaging's
`/orcd/pool/005/cge7/tandem-rlvr`. Engaging has matching dataset/grader hashes,
identical patched vLLM/verl source trees and identical compiled vLLM libraries.
The original failed calibration and interrupted runs are preserved.
CSAIL scratch was full and its separate project quota had 784 GiB remaining while
the existing Qwen/elegance runs were still growing. The new full jobs use Engaging
pool storage. Its untyped GPU/feature request was rejected; the valid typed H200
request includes every H200 in `mit_preemptable`, the only GPU class large enough
for this measured configuration. No racing or duplicate continuation was submitted.

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
steps and full validation. The queued GRPO profile was interrupted during startup
to combine its remaining timing check with a checkpoint-writing smoke. The cache calculation uses the actual
4,096-prompt + 6,000-response cap, not the unused remainder of the 12,288-token
model context. At this stage full training remained stopped pending calibration
admission and a complete-run forecast below 48 GPU-hours; the relaunch section
above supersedes that status.
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
