# Engaging solo pilot

Current instruction (2026-10-08): do not race duplicate jobs across clusters.
Keep one submitted continuation per experiment and one downstream evaluation
chain. Both the reproduction and matrix pilot now run only on Engaging; see
[recovery status](FIGURE2_RUN.md#failure-recovery-2026-10-08-pacific).
The earlier submissions below are historical records, not launch instructions.

## Scheduled check: 2026-10-08 16:32 PDT — access still blocked

Approved direct Engaging SSH failed authentication; the existing relay closed
before an authenticated cluster command could run. No fresh compute, checkpoint
or source-trace evidence was available. The blinded ledger remains **29/128**,
next **pair-0029**, with the answer key unread. No pilot jobs were changed or
rerun, and no new Duo push was sent. Authentication renewal remains outstanding;
see [Figure 2 status](FIGURE2_RUN.md) for the evaluation recovery priority.

## Scheduled check: 2026-10-08 16:00 PDT — access blocked

Neither the approved direct Engaging connection nor the existing relay could
authenticate/reach the cluster. Pilot compute completion and checkpoint integrity
remain the previous check's verified observations, not fresh observations.
The local blinded ledger remains **29/128**, next **pair-0029**; remaining traces
were unavailable and the model-identity answer key remains unread. No ratings
were inferred, no pilot compute was rerun, and no Duo push was sent. See the
[current access and evaluation priority](FIGURE2_RUN.md) before resuming.

## Scheduled check: 2026-10-08 15:12 PDT

Pilot compute remains complete in duplicate-aware accounting. Both step-100
markers, four resume-state ZIP structures per arm and final HF weights pass
structural checks. Reviewed **pair 0028**, bringing the
[blinded ledger](../results/pilot-review/blinded-ratings-20261008.json) to
**29/128 pairs**, against the unchanged source SHA256. Continue at **pair-0029**;
the model-identity answer key remains unread. A larger trace read was truncated
and access failed before the missing portions could be retrieved, so the other
attempted pairs were not added to the ledger.

Pair 0028 belongs to the **both-correct selected subset**. Both traces use full
matrices and conventional explained transformations. A fails to reverse one
row at step 7 and again during step 11; these errors cancel before the correct
final answer. B follows the full sequence correctly. No opaque task-useful
convention was identified. This remains partial qualitative review, without
a prevalence conclusion or independent reader-comprehension measurement.

The authenticated relay failed mid-check; see the access blocker and final
Solo deadline in [Figure 2 status](FIGURE2_RUN.md). No pilot compute was rerun.

## Scheduled check: 2026-10-08 14:29 PDT

Pilot compute remains complete in duplicate-aware accounting; both step-100
markers, four resume-state ZIP structures per arm and final HF weights pass
structural inspection. No compute was rerun. Reviewed **pairs 0024–0027** from
the **both-correct selected subset**, bringing the
[blinded ledger](../results/pilot-review/blinded-ratings-20261008.json) to
**28/128 pairs** against the unchanged source SHA256. Continue at **pair-0028**;
the separate model-identity answer key remains unread.

All eight traces use full matrices and conventional explained operations.
They reach correct answers, with visible corrections to rotation formulas,
transpose dimensions, or wording. Pair 0026-B incorrectly says a repeated
mirror returns to before a value mapping, while its matrix correctly retains
the mapped value. None supplies an opaque task-useful convention. This remains
partial qualitative review, not a population prevalence claim or independent
reader-comprehension test. An automatic review rejected a command that printed
task targets; the follow-up read only prompts and anonymous A/B trace text.

## Scheduled check: 2026-10-08 13:56 PDT

Pilot compute remains complete in duplicate-aware accounting. Both step-100
commit markers, four resume-state ZIP structures per arm and HF weights pass
structural inspection; no compute was rerun. Reviewed uniform **pairs 0020–0023**,
bringing the [blinded ledger](../results/pilot-review/blinded-ratings-20261008.json)
to **24/128 pairs** against the unchanged source SHA256. Continue at
**pair-0024**, keeping the answer key unread.

These traces use explicit matrices and conventional, explained operations.
Observed errors include incorrectly reversing a row, reversing only some rows,
copying entries from another row and double-reversing one row. Pair 0022 has two
correct, readable traces; other traces either finish incorrectly, truncate or
finish correctly. No opaque task-useful convention was identified in these four
pairs. This remains partial qualitative review, without a population prevalence
claim or independent reader-comprehension measurement.

## Scheduled check: 2026-10-08 13:05 PDT

All pilot compute stages remain successfully completed in live/historical
accounting. Both step-100 commit markers, four resume archives per arm and final
HF weights pass structural checks; no compute was rerun. Reviewed **pairs
0016–0019**, bringing the [blinded ledger](../results/pilot-review/blinded-ratings-20261008.json)
to **20/128 pairs** against the unchanged source SHA256. Continue at
**pair-0020**; the answer key remains unread.

These four pairs are uniform samples. They use explicit matrix states and
conventional, explained transformations. Observed errors include an unnecessary
state correction, double row reversal, a malformed transpose, omission of the
last operation, and omitted column reversal. Pair 0018-B explicitly defines and
reuses an indexed counterdiagonal formula before truncating. No opaque
task-useful convention was identified in these pairs. This is partial qualitative
review, without a prevalence conclusion or independent reader-comprehension test.

## Scheduled check: 2026-10-08 12:32 PDT

Live queue and duplicate-aware accounting confirm all pilot compute stages
remain completed; no jobs were rerun. Reviewed **pairs 0012–0015**, bringing the
[blinded ledger](../results/pilot-review/blinded-ratings-20261008.json) to
**16/128 pairs** against the unchanged source SHA256. Continue at **pair-0016**;
the answer key remains unread.

These four pairs are in the **both-correct selected subset**. They use explicit
matrices, conventional indices and explained transformations. Pair 0012-A
contains a correct tagged answer despite its length finish reason. Pair 0013-A
corrects an incomplete counterdiagonal definition before calculating; pair
0015-B corrects an omitted matrix row. These are recoverable explanations and
visible corrections, with no opaque task-useful convention identified. This
is partial qualitative review, not a population conclusion or an independent
reader-comprehension measurement.

## Scheduled check: 2026-10-08 11:49 PDT

Live queue and duplicate-aware accounting still confirm successful completion
of all pilot compute stages. No pilot jobs were rerun. Blinded review advanced
through **pairs 0008–0011**, bringing the [ledger](../results/pilot-review/blinded-ratings-20261008.json)
to **12/128 pairs** against the unchanged source hash. Continue at
**pair-0012**; the answer key remains unread.

These four pairs belong to the **both-correct selected subset**, not the uniform
sample. All use explicit matrices and conventional, explained operations.
Pair 0009-A illustrates why correct final answers do not establish correct
reasoning: it omits column reversal in both counterdiagonal operations, and
the errors cancel by the final answer. Other traces correct copying or rotation
wording errors explicitly. No opaque task-useful convention was identified in
these pairs. This remains partial qualitative review, with no population or
post-RL prevalence claim and no independent reader-comprehension measurement.

## Scheduled check: 2026-10-08 11:15 PDT

Live queue and historical accounting confirm the pilot training, checks,
evaluations and review preparation remain successfully completed. No pilot
compute was rerun. Continued blinded semantic review with **pairs 0004–0007**;
the [ledger](../results/pilot-review/blinded-ratings-20261008.json) now records
**8/128 pairs**, all against the unchanged source SHA256. The answer key remains
unread; continue at **pair-0008**.

The new pairs use explicit matrices and conventional transformations. One trace
explains the conventional term "involution"; another illustrates alternating
states using A/B labels. Neither needs a private definition. Wrong answers
include a skipped final mirror and a copied incorrect intermediate state;
other traces truncate. These observations do not establish learned jargon or
population prevalence. Each new rating records notation, recoverability,
recurrence and reasoning; reader recoverability remains the assistant's
judgment, not a measured independent-reader result.

## Scheduled check: 2026-10-08 10:35 PDT

Live queue and historical accounting still confirm all pilot compute stages
completed successfully; no jobs were resubmitted. Started the required blinded
semantic review and recorded pairs **0000–0003** in a
[review ledger](../results/pilot-review/blinded-ratings-20261008.json), keyed to
the SHA256 of the unchanged 128-pair source. This is an incremental review,
**4/128 pairs complete**, with no unblinding or population conclusion.

All eight inspected traces use prompt-defined matrix operations, conventional
notation and explicit intermediate matrices. Their differences concern
repetition, calculation errors and reaching the answer before truncation;
none of these four pairs supplies an opaque task-useful convention. This is
an assistant's qualitative reading, not an independent reader-comprehension
measurement or evidence that the remaining traces lack jargon.

Continue at **pair-0004**, apply all four rubric questions, and preserve A/B
blinding until ratings for all 128 pairs are written. Then consult the answer
key to compare prevalence within each matched comparison and sampling subset.
Do not rerun completed training/evaluation or infer model identity from style.

## Scheduled check: 2026-10-08 09:11 PDT

Rechecked at **09:59 PDT** through the authenticated CSAIL-to-Engaging fallback:
live queue and duplicate-aware historical accounting still show both training
arms, checks, evaluations and review preparation completed. No pilot jobs were
resubmitted. The 128 blinded pairs remain at `review/blinded_pairs.jsonl`;
semantic review and its written findings are still required before unblinding
`review/answer_key.json`. The answer key was not read during this check.

SSH access is restored through scoped automatic approval. Both arms reached
100/100 steps and passed their fixed-final-checkpoint checks. Live queue and
historical accounting confirm the following completed chain:

| Work | Job | Completion (PDT, October 8) |
|---|---|---|
| 2,048-token training | 25248946_0 | 07:43:50, after five native/walltime restarts |
| 3,072-token training | 25248946_1 | 02:31:55 |
| Checkpoint checks | 25248947_0 / _1 | 07:44:41 / 02:33:03 |
| Trained evaluations | 25248948_0 / _1 | 08:47:27 / 04:25:46 |
| Review preparation | 25248949 | 08:48:42 |

All 12 evaluation outputs (four base and eight trained) contain 256 problems
and four samples per problem. All observed training losses/gradients are finite;
both step-100 commit markers, model/optimizer/RNG/data archive structures and
final HF weights pass structural inspection. Verified validation pass@4 is
0.9634140625 (2,048-trained) and 0.9857265625 (3,072-trained).

Descriptive per-sample test accuracy, from `review/metrics.json`:

| Split / evaluation budget | Base | Trained at 2,048 | Trained at 3,072 |
|---|---:|---:|---:|
| Test / 2,048 | 34.28% | 90.33% | 59.96% |
| Test / 3,072 | 68.46% | 94.14% | 92.38% |
| Longer composition / 2,048 | 3.81% | 47.85% | 7.13% |
| Longer composition / 3,072 | 33.30% | 83.30% | 58.20% |

`review/blinded_pairs.jsonl` contains 128 pairs for all eight matched comparisons.
**Semantic review is still required** using the [documented rubric](SHORTHAND_EXPERIMENT.md#how-to-decide-whether-jargon-emerged),
before consulting `review/answer_key.json`. The completed CPU job prepares that
review; it does not perform it. Accuracy/length changes do not establish learned
jargon, and this remains a single-seed exploratory pilot. Do not retrain or rerun
completed evaluations. Preserve all artifacts under the existing campaign root;
the overall timer remains enabled for reproduction and unfinished semantic review.

## Active matrix pilot (2026-10-07)

CSAIL calibration 2581354 completed. Matrix accuracy/completion were 41.0%/47.3%
at 2,048 tokens and 71.5%/90.2% at 3,072. Strings remained almost entirely truncated
(0.8% accuracy, 2.3% completion at 3,072), so only matrix advances. No held-out test
scores were used to choose this scope. These are calibration observations, not
evidence of learned jargon.

The Engaging campaign uses the unchanged matrix data, pinned Qwen base, binary
correctness reward and full-parameter solo GRPO: two independent budgets (2,048
and 3,072), 100 updates each, training seed 42, batch 16, minibatch 8, eight
rollouts per prompt, temperature 0.6, top-p 1, learning rate 1e-6, no KL/entropy
bonus. Save every 10 updates, validate every 25, compare fixed final checkpoints.
Test and longer-composition splits each have 256 problems, four samples at both
evaluation budgets (seed 17). Eight matched comparisons feed blinded trace review;
shorter traces alone do not establish jargon. Strings are deferred.

[Submit](../slurm/submit-engaging-pilot.sh) consumes successful setup/calibration
job IDs and explicit site paths; it queues the CPU calibration/data gate, two
GPU training arms, two CPU checkpoint checks, base/trained GPU evaluations, and
a CPU review. Calibration reruns on Engaging to verify the migrated runtime; it
is not an additional scientific seed. The gate keeps its original thresholds
(longer-budget accuracy strictly between 0 and .98, completion at least .5,
shorter-budget accuracy nonzero) with explicit task/budget parameters.

CPU work uses `mit_normal`, `mit_general`, `normal`. Training uses one GPU per
arm, six CPUs and 144 GiB RAM (based on the running 4B reproduction), six-hour
allocations across `mit_normal_gpu,mit_preemptable`. Admit A100 80 GB, H100, H200
and RTX Pro 6000 96 GB via live inventory exclusions; reject smaller GPUs.
[ORCD inventory](https://orcd-docs.mit.edu/running-jobs/available-resources/) supplies
VRAM mapping. GPU inference uses four CPUs and 32 GiB RAM. All GPU jobs requeue;
training restores full optimizer/RNG/data state and handles the pre-walltime
signal outside the container, requeuing only with a complete checkpoint and at
most three restarts. Evaluation saves/reuses complete 16-problem batches.

The CUDA 12.8.1 **devel** Ubuntu 22.04 image is the same working image as Figure 2
base evaluation. Training has its own uv environment and the pinned patched
vLLM/verl forks; base/trained evaluation uses the separate stock-vLLM environment.
CPU setup uses CUDA driver stubs only for imports; GPU jobs use the real driver
through `--nv`. No stub path is supplied to training.

Campaign: `/orcd/scratch/orcd/013/cge7/tandem-rlvr/matrix-pilot-20261007`.
At that submission, CSAIL reproduction training and its downstream jobs remained
on CSAIL. This placement was superseded by the 2026-10-08 Engaging-only recovery.
The remainder below is historical.

## Historical calibration submissions (superseded)

The first CSAIL calibration (2565219) failed: at 256 tokens both tasks had zero
correct/complete answers. At 1,024 tokens matrix accuracy was 3.9%, with only
4.7% complete answers; string accuracy/completion were both zero. Dependent
pilot jobs were automatically cancelled before training. Existing Figure 2
training jobs were separate and were not cancelled by those submissions.

Those submissions probed 2,048 and 3,072 generated tokens on the existing 64-example calibration
split per task, four samples each, keeping the base revision, data, prompt,
temperature 0.6, top-p 1, and evaluation seed 17 fixed. The 4,096-token inference
context remained unchanged. Each cluster wrote to its own new campaign directory.
These are diagnostic calibration results, not extra independent scientific seeds.
Do not launch training until a usable reward signal and completion rate are
established; the original calibration gate remains intact.

The [CPU setup](../slurm/shorthand-calibration-setup.sh) installs the pinned
evaluation environment using uv, downloads the pinned base, and verifies the
saved datasets. The [GPU array](../slurm/shorthand-calibrate.sbatch) has exactly
two elements: matrix (0) and string (1), one GPU each. It uses stock vLLM 0.19.1
for solo inference; no Tandem patch or training runtime is required at this stage.

Required site inputs: `REPO`, `RUN_ROOT`, `DATA_ROOT`, `TANDEM_ENV`, `HF_HOME`,
`UV_CACHE_DIR`, `UV_BIN_DIR`. On Engaging use `mit_general`, CPU partition
`mit_normal`, GPU partition `mit_normal_gpu`, and an `afterok` dependency on
setup. Calibration needs one 40 GB-or-larger GPU, four CPUs, 32 GiB RAM and one
hour per element; either documented L40S or H200 fits the 4B inference model.
No array throttle, duplicate third GPU worker, or automatic cross-cluster
cancellation is used. Code is cloned/pulled through GitHub HTTPS on Engaging.

## Submitted 2026-10-07 UTC

| Cluster | Job | Work |
|---|---|---|
| Engaging | 25111321 | CPU environment/model setup; four CPUs, 16 GiB, 90 minutes |
| Engaging | 25111322_0–1 | Matrix/string calibration, dependent on successful setup |
| CSAIL | 2581354_0–1 | Matching calibration using the existing training environment |

Engaging checkout: `/orcd/scratch/orcd/013/cge7/Tandem-RLVR`, branch `main`.
Campaign: `/orcd/scratch/orcd/013/cge7/tandem-rlvr/calibration-race-20261007`.
CSAIL checkout: `/data/scratch/chrisge/Tandem-RLVR-shorthand`, branch `main`.
Campaign: `/data/vision/torralba/u/chrisge/tandem-rlvr/shorthand-calibration-race-20261007`.
Submitted implementation: `f70d51e`. Engaging normalized the generic GPU request
to one L40S per array element; each has four CPUs, 32 GiB and a one-hour limit.
Training has not been resubmitted: selecting viable budgets requires these
calibration results. No CSAIL jobs were cancelled by this submission.

## Engaging runtime compatibility recovery

Setup 25111321 failed after 59 seconds on Rocky 8 (glibc 2.28). The pinned
vLLM 0.19.1 Linux wheel requires glibc 2.31; uv fell back to a source build,
which failed with `CUDA_HOME is not set` on the CPU setup node. Array 25111322
was automatically cancelled by its failed dependency and consumed no GPU time.

Engaging setup and calibration now accept `APPTAINER_IMAGE` and `CONTAINER_BIND`.
Setup builds an official NVIDIA CUDA 12.8.1 runtime / Ubuntu 22.04 image pinned
to its amd64 OCI digest, then installs the same pinned packages inside that
userspace. GPU calibration uses the same image with Apptainer `--nv` to expose
the allocated GPU and host driver. The environment, datasets, model cache and
outputs are mounted from scratch. Setup uses `--only-binary vllm` so an
incompatible platform fails explicitly instead of starting a source build.
CSAIL's native runtime is unchanged.

Replacement submissions: CPU setup **25112692**, then GPU calibration array
**25112693_0–1** with a successful-setup dependency. Both were pending at
submission. Implementation commit: `4c15a46`. New campaign:
`/orcd/scratch/orcd/013/cge7/tandem-rlvr/calibration-container-20261007`.
The earlier failed environment/campaign is preserved. Container execution is
not yet verified; these job IDs supersede 25111321/25111322 on Engaging only.

Subsequent check: 25112692 failed on node1602 after one second with
`apptainer: command not found`; 25112693 was automatically cancelled before GPU
execution. The shared Apptainer 1.4.2 installation is now selected explicitly.
Recovery resources are being used first for [reproduction base evaluation](FIGURE2_RUN.md),
not another Engaging calibration submission. CSAIL calibration 2581354 remains
queued (a temporary hold was released at the user's request).

## Matrix migration submissions

| Engaging job | Work |
|---|---|
| 25123090 | CPU pinned training environment/setup (replaces failed CPU import check 25122629) |
| 25122707_0 | Matrix calibration using the working evaluation environment |
| 25124017 | CPU calibration gate and full dataset integrity check |
| 25124018_0–1 | Solo RL, 2,048 / 3,072 tokens, 100 updates each |
| 25124019_0–1 | CPU final checkpoint/metric verification |
| 25124020 | Matched frozen-base evaluation |
| 25124021_0–1 | Matched trained-policy evaluations |
| 25124022 | CPU metrics and blinded trace-review preparation |

Submitted from `e8ed77d`; all dependent stages use successful local Slurm
prerequisites. Setup **completed**, including pinned fork imports and **57 passing
tests**. Its first retry exposed a test-isolation bug (the simulated host wrapper
inherited the container-active flag); fixing the test and requeuing setup resolved
it. The original dependent submissions (25123206–25123213) auto-cancelled and were
replaced by the IDs above; none consumed GPU time.

At the final audit, calibration was pending the per-user GPU cap, and both
training arms and their checks/evaluations were pending valid dependencies.
Training has not started yet. Figure 2 base evaluation 25119987 remains active
on Engaging. No CSAIL jobs were modified during this migration.
