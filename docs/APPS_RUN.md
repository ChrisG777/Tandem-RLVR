# APPS pilot run: 2026-10-10

Status: implementation and interactive inference checks are complete; full
validation calibration is running. Training, checkpoint selection, final test
evaluation and scientific plots have **not** completed. This remains a single-seed
pilot, governed by the [APPS protocol](APPS.md).

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

The source dataset revision is `21e74ddf8de1a21436da12e3e653065c5213e9d1`.
Preparation retained 3,890 training problems, 200 validation problems, and all
4,000 introductory/interview test problems (1,000 introductory; 3,000 interview).
The audit excluded 549 training problems with empty/malformed tests. No retained
test problems exceed the prompt limit. A valid large-integer test was preserved
by disabling Python 3.11's integer-string digit cap inside the bounded runtime.

Solo calibration was deliberately interrupted after a completed batch to test
resumption. The saved texts are retained; the final restart receipt verifies that
the completed run preserves all 16 saved problem records. Native-word calibration
and the gate follow in the same allocation. A failed gate stops training; passing
the gate admits independent three-step GRPO/Tandem smoke runs before full training.
