# Figure 2 attempt: status 2026-10-05 (America/Los_Angeles)

## Scheduled check: 2026-10-08 12:32 PDT

**Both reproduction arms have completed 200 steps and final validation.**
Tandem **25248795** completed successfully at **12:10:38 PDT**; selection
**25248797** completed at **12:11:23 PDT**. Live queue/control and duplicate-aware
historical accounting agree. Re-running the existing verifier passes both arms;
Tandem's documented missing metric steps 20 and 160 remain explicit, not imputed.
Both step-200 commit markers, four resume-state ZIP structures per arm, and final
HF safetensors pass structural checks (not a full tensor reload).

| Arm | Best-validation selection | Final step-200 validation pass@4 |
|---|---|---|
| Solo GRPO | Step 160, 67.3520% | 67.2548% |
| Tandem | Step 180, 65.6210% | 64.4738% |

The generated `models.json` selects these verified paths. Its resolved GRPO
checkpoint differs from `models-final-grpo.json` step 200, confirming the two
evaluation selections are distinct. Keep both chains. The existing full Solo
curves already include final validation and require no refresh. Tandem's final
senior-token fraction is 0.505932; observed losses/gradients pass finite checks.

At **12:29 PDT**, final jobs **25299322/25299323** have saved **560 solo / 336
handoff problems**. Best-validation jobs **25248799/25248800/25248802** began at
12:12:23 PDT and have saved **48 GRPO solo / 48 Tandem solo / 16 GRPO handoff
problems**. All saved identities are unique, sample counts are 32 solo / 8
handoff, and unfinished-chain counts are zero. These are partial counts, not
final benchmark scores. Recent logs show no new application errors.

Tandem handoff **25248803** is eligible and waiting on `QOSMaxGRESPerUser`;
plot **25248804** retains successful dependencies on all four best-validation
evaluations. No jobs were replaced, requeued manually, or duplicated. All five
running evaluations have zero restarts. The final jobs' current six-hour limits
end at **15:16:35 PDT**; check progress before then and, if continuation is needed,
preserve complete batches and job dependencies. The launcher resumes batches on
restart but does not install a pre-walltime requeue trap; native `--requeue`
alone does not handle timeout. Do not launch a concurrent replacement.

Quota timestamp **12:20 PDT**: scratch **785.6/1024 GB**, pool **666.5/1024 GB**,
home **103.4/200 GB**. Reported headroom is adequate subject to report lag.
Direct SSH still fails authentication after scoped escalation; the authenticated
CSAIL-to-Engaging relay works. No Duo push or CSAIL job action occurred.

Pilot compute remains complete, and blinded review now covers **16/128 pairs**;
continue at **pair-0016**, with the answer key unread. Keep the existing timer
enabled for both evaluation selections, Figure 2 export and remaining review.

## Scheduled check: 2026-10-08 11:49 PDT

Tandem **25248795** remains in final step-200 validation on node5106,
restart 2. Metrics remain through 199, with finite observed losses/gradients
and senior-token fraction 0.499651. Its log shows continuing validation data
activity through 11:44 PDT; there is no new application failure. The step-200
commit marker, four resume-state ZIP structures and HF safetensors pass
structural inspection for both reproduction arms. Successful final validation
and job exit are still required for Tandem completion.

Solo remains complete. Re-running the verifier selects **step 160**, whose
resolved path differs from final manifest **step 200**. Full Solo curves already
contain the final validation record. At 11:49 PDT final evaluations
**25299322/25299323** have saved **512 solo / 176 handoff problems**. All saved
identities are unique, with 32 / 8 samples per problem and zero unfinished
chains. Both jobs are running without restarts; these are partial counts.

Live queue/control and duplicate-aware historical accounting agree. Selection
**25248797** still waits on Tandem; all four best-validation evaluations and
plot **25248804** retain their successful dependencies. No jobs were replaced
or duplicated. Confirm the generated GRPO manifest path matches the verified
step-160 selection when selection finishes.

Quota timestamp **11:19 PDT**: scratch **785.6/1024 GB**, pool **666.3/1024 GB**,
home **103.4/200 GB**. Headroom remains adequate subject to report lag. Direct
SSH fails authentication after scoped escalation; the authenticated
CSAIL-to-Engaging relay works. No Duo push or CSAIL job action occurred.

Pilot compute remains completed. Blinded review now covers **12/128 pairs**;
continue at **pair-0012**, keeping the answer key unread. The timer remains
enabled for final Tandem validation, both evaluation selections, Figure 2
export and remaining semantic review.

## Scheduled check: 2026-10-08 11:15 PDT

Tandem **25248795** has committed **step 200** and is running final validation
on node5106 (restart 2). Metrics are complete through **199/200**, with finite
observed losses/gradients and step-199 senior-token fraction **0.499651**.
The step-200 model, optimizer, extra-state and dataloader ZIP structures and HF
safetensors pass structural checks. Final validation and successful job exit
remain necessary before declaring training complete; no partial save was promoted.

Solo remains complete, and the existing verifier again selects **step 160**,
distinct from the final manifest's **step 200**. Its full curves already contain
the final validation record and need no refresh. At **11:14 PDT**, final jobs
**25299322/25299323** have **400 solo / 128 handoff problems** saved, respectively.
Saved identities are unique, each has the expected 32 / 8 samples, and both
progress files report zero unfinished chains. Both jobs remain running without
restarts; these counts are partial evaluation progress, not final scores.

Live queue/control and historical `sacct -D` agree. Selection **25248797** still
waits on Tandem, all four best-validation evaluations wait on selection, and
plot **25248804** waits on those evaluations. No new failure, replacement or
dependency repair was needed. Confirm the generated manifest's GRPO path is
the verified step-160 path before using the best-validation outputs; the
step-160 and step-200 evaluations are distinct work.

Quota report timestamp **10:49 PDT**: scratch **785.6/1024 GB**, pool
**651.7/1024 GB**, home **103.4/200 GB**. Reported headroom remains adequate,
subject to report lag. Direct SSH fails authentication after scoped escalation;
the existing authenticated CSAIL-to-Engaging relay works. No Duo push was sent.

Pilot compute remains completed. Blinded review now covers **8/128 pairs**;
continue at **pair-0008** without opening the answer key. The timer remains
enabled for final Tandem validation, both evaluation selections, Figure 2
export and the remaining semantic review.

## Scheduled check: 2026-10-08 10:35 PDT

Live `squeue`/`scontrol` and duplicate-aware `sacct -D` agree: Tandem
**25248795** is running on node5106, restart 2, with metrics through **193/200**
and committed resume **step 190**. Observed losses/gradients remain finite;
step-193 senior-token fraction is 0.504316. The step-190 model, optimizer,
extra-state and dataloader ZIP structures pass inspection; the most recent
validation HF weights (step 180) pass the existing safetensors verifier.
These are structural checks, not a full tensor reload.

Solo remains successfully complete. Re-running the existing verifier selects
**step 160**, whose resolved path differs from final manifest **step 200**.
Final jobs **25299322/25299323** are running without restarts and have saved
**160 solo / 64 handoff problems**, respectively, at 10:35 PDT. All saved
identities are unique and have the required 32 / 8 samples. Neither evaluation
is complete. The completed Solo curves already contain step-200 validation;
no new training record requires another plot refresh.

Selection **25248797** waits only for Tandem. The four best-validation
evaluations still depend on successful selection, and **25248804** depends on
all four evaluations. No job replacement or dependency repair is needed.
Before consuming the generated manifest, confirm its GRPO entry still matches
the verified step-160 selection; retain the final/best distinction.

Quota report timestamp **10:19 PDT**: scratch 785.6/1024 GB, pool 651.7/1024 GB,
home 103.4/200 GB. Reported headroom remains adequate, allowing for report lag.
Direct SSH still fails authentication after scoped sandbox escalation; the
existing authenticated CSAIL-to-Engaging relay succeeds. No Duo push was sent.

Pilot compute stages remain completed. Blinded semantic review has begun:
four of 128 pairs are recorded in the [review ledger](../results/pilot-review/blinded-ratings-20261008.json).
The answer key remains unread and no learned-jargon conclusion is available.
Keep the existing timer enabled for training, both evaluation selections,
Figure 2 export and the remaining semantic review.

## Scheduled check: 2026-10-08 09:59 PDT

Solo **25248796 completed successfully at 09:29 PDT**, including final
validation. The existing verifier passes all 200 logged steps with no missing
or nonfinite observed metrics. Final step-200 validation pass@4 is
**67.2548%**; best validation selects **step 160, 67.3520%**. Its resolved
checkpoint path differs from `models-final-grpo.json` (step 200), so the
best-validation and final evaluation chains are distinct and should remain.
Recheck the generated selection manifest before acting on downstream outputs.

The [Solo curves](../results/training-progress/solo-20261008.png), SVG, summary
and timestamped snapshot now include all 200 steps and final validation
(snapshot 09:57 PDT; results commit `9c81ca6`). Plotting inside the CUDA image
required mounting the host `/usr/share/zoneinfo` read-only at the same path;
no training/runtime dependency or scientific setting changed.

| Work | Live evidence at 09:56–09:59 PDT |
|---|---|
| Tandem 25248795 | Running on node5106, restart 2; metrics through 186/200, committed resume step 180. Step-180 validation pass@4 65.6210%; recent losses/gradients finite, senior-token fraction 0.5005–0.5060. |
| Final Solo eval 25299322 | Running on node3006 (L40S) since 09:16:35; 64 complete problems saved, 32 samples each. |
| Final handoff eval 25299323 | Running on node3203 (L40S) since 09:16:35; 32 complete problems saved, 8 samples each. |
| Selection 25248797 | Waiting only for Tandem; Solo dependency satisfied. |
| Best-validation evals / plot | Original IDs and successful dependencies intact; no duplicates or replacements submitted. |
| Pilot | Both training/check/evaluation chains and review preparation remain completed in accounting; blinded semantic review remains outstanding. |

Both reproduction commit markers, four required resume-state files per arm,
ZIP structures and latest retained HF safetensors pass structural checks.
Saved final-evaluation batches have unique problem identities and the expected
sample counts. This is not a full tensor reload or final benchmark completion.
Historical `sacct -D` and live `squeue`/`scontrol` agree; no new application
failure was found. Quota report timestamp **09:48 PDT**: scratch 785.6/1024 GB,
pool 650.2/1024 GB, home 103.4/200 GB, leaving adequate checkpoint headroom.

Direct `ssh orcd-login` now lacks its local ControlMaster and fails
keyboard-interactive authentication even after scoped sandbox approval. The
documented **CSAIL-to-Engaging ControlMaster works**; use that fallback for
subsequent checks while it remains authenticated. CSAIL emits an AFS `.bashrc`
permission warning, but the scratch/pool-based Engaging commands succeed.
No Duo push, authentication bypass, security change or CSAIL job action occurred.
The existing timer remains enabled.

## Solo final-checkpoint evaluation: 2026-10-08 09:16 PDT

The user requested full Solo curves and evaluation of its final model. The
step-200 commit marker, dataloader state, model/optimizer/extra-state files and
HF safetensors structure were verified; training was still finishing its final
validation, so the monitoring snapshot has all steps 1–199 and validation through
180. [Solo curves](../results/training-progress/solo-20261008.png),
[summary](../results/training-progress/solo-20261008.json) and the timestamped
source snapshot are under `results/training-progress/`.

Independent Engaging jobs **25299322** (`grpo-solo`, 32 samples/problem) and
**25299323** (`grpo-handoff`, 8 samples/problem) evaluate immutable step-200
weights. Both were submitted with no training dependency and were pending
priority at the submission check. They use one compatible GPU, four CPUs,
48 GiB RAM, six hours, native requeue and per-batch evaluation progress across
`mit_normal_gpu,mit_preemptable`; only incompatible L4 nodes are excluded.
The existing decoding settings, benchmark splits and frozen base are unchanged.

Manifest: recovery `reproduction/models-final-grpo.json`; outputs:
`reproduction/eval-final-grpo/grpo/{solo,handoff}.json`; logs:
`logs/final-grpo-{solo,handoff}-JOB.out`. This is **final-checkpoint** evaluation,
distinct from the still-pending best-validation reproduction chain. Preserve
that distinction in comparisons. If best validation selects step 200, reuse the
matching completed final evaluation instead of performing duplicate work. The
user was asked whether to retain the best-checkpoint comparison as well; no
answer had arrived at submission, so its existing dependent jobs were unchanged.

The evaluation launcher now accepts manifests containing only the base and the
arm required by its phase. Per-job model-path files avoid concurrent phases
overwriting one another. Shell syntax and diff checks passed; the final jobs'
GPU execution is pending.

## Scheduled check: 2026-10-08 09:11 PDT

The authenticated `orcd-login` connection works with scoped automatic approval;
the 09:03 sandbox failure below is superseded. No Duo renewal was needed.
Engaging remains the sole execution cluster. No job IDs were replaced.

| Work | Live evidence |
|---|---|
| Tandem 25248795 | Running on node5106, two RTX Pro 6000 GPUs, restart 2. Committed step 180/200; metrics through 179 while validation runs. |
| Solo 25248796 | Running on node4004, one RTX Pro 6000 GPU, restart 1. Committed step 200/200; metrics through 199 while final validation runs. This is not yet verified completion. |
| Selection 25248797 | Valid `afterok` dependencies on both training jobs. |
| Evaluations 25248799/25248800/25248802/25248803 | Valid `afterok` dependency on selection. |
| Plot 25248804 | Valid `afterok` dependencies on all four evaluations. |
| Matrix pilot | Both 100-step arms, checks, evaluations and review preparation completed; semantic review remains outstanding. See [pilot results](ENGAGING_PILOT.md#scheduled-check-2026-10-08-0911-pdt). |

Historical `sacct -D` preemptions and the 05:54:33 Tandem allocation ending in
REQUEUED agree with live restarts; they are not new application failures.
Tandem's current load log confirms restoring model, optimizer, RNG and scheduler
from step 160. Recent observed losses/gradients are finite; step-179 senior-token
fraction is 0.50588. At 09:10 PDT both Tandem GPUs were active (36%/33% utilization).
The long step-180 validation has not emitted its final metric yet; inspect its
progress at the next check rather than claiming a completed step from the save.

Checkpoint audit: latest committed model/optimizer/extra-state/dataloader files
are present with readable ZIP structures for both reproduction and both pilot
arms. Retained validation HF files pass the existing safetensors checks. These
are structural checks, not a full tensor reload/bytewise checksum. The old Solo
`hf/global_step_130` from the quota failure remains invalid and preserved; it is
not a validation candidate or a current resume checkpoint. Historical
`selected.json` files still describe step 120 and must not be used as final
selections.

Fixed a forthcoming verification failure in `f5ce275`: resume-evidence matching
now accepts a logged absolute storage alias only when it resolves to the exact
expected checkpoint file. Eleven focused tests pass locally and on Engaging,
including rejection of an alias redirected to another run. Pushed to reproduction
`main` and deployed by Git fast-forward. Recorded
`tandem-full/resume-evidence/step-160.json` from the actual four load messages
and adjacent attempts. Step-160 metrics/validation remain explicitly missing
and are excluded from selection; no values were imputed. The imported step-20
gap remains covered by its migration receipt.

Quota report timestamp **08:47:18 PDT**, observed at 09:05 PDT: scratch
785.3/1,024 GB (76.68%), pool 635.4/1,024 GB (62.05%), home 103.4/200 GB.
This leaves reported scratch/pool margins of 238.7/388.6 GB, subject to report
lag and intervening writes. No data was deleted or moved during this check.
Keep the existing timer enabled: reproduction training/selection/evaluation/export
and pilot semantic review remain incomplete.

## Scheduled check: 2026-10-08 09:03 PDT — access blocked

The independent scheduled monitor found no `latest-status.md` in
`~/Library/Logs/tandem-rlvr-monitor/`. It read the campaign documents and shared
allocation/cluster guides, then tried `ssh orcd-login` with batch authentication.
The session sandbox denied access to the local ControlMaster socket
(`Operation not permitted`); hostname resolution also failed. The authorized
CSAIL-to-Engaging fallback could not reach CSAIL because its local ControlMaster
socket was likewise denied. Neither attempt reached a cluster command.

Live queue/accounting, training metrics, checkpoint integrity, dependencies and
the timestamped scratch/pool quota report are **unverified at this check**. The
recovery status below is historical evidence, not a fresh observation. No jobs,
experimental data, authentication settings or timer configuration were changed;
no Duo push was sent. Authentication expiry has not been established. Completion
cannot be established, so the existing timer remains enabled. Cluster inspection
requires a session with permitted access to the existing SSH connections; do not
infer a training failure or submit replacements from this access failure.

At that check this entry remained local pending Git publication. Scoped automatic
approval restored access at the next check above; both records are now published.

## Failure recovery: 2026-10-08 Pacific

**Current execution: Engaging only.** At 00:32 PDT the user ended the earlier
cross-cluster race. Earlier race permission is withdrawn: keep one submitted
continuation per experiment and one downstream chain across clusters.
CSAIL training 2596961/2596962 and downstream jobs
2596963–2596968 were cancelled and verified cancelled in Slurm; their files are
preserved. They were redundant continuations of the same experiments from step
120, not independent replications. Keep Engaging training 25248795/25248796 and
its evaluation chain. The separate shorthand pilot remains on Engaging.
At 00:33 PDT Solo was running; Tandem had been preempted on node2100 and
automatically requeued (restart 1), without an application traceback.

The previous Engaging Solo job reached its step-130 save and failed with
`EDQUOT`; `data.pt` and the commit marker were not written, so it must resume
from 120. Engaging scratch reached its 1,024-GB user quota despite 252 TB free
on the shared filesystem. Initial profiling found 482 GiB in Tandem/pilot
artifacts, 341 GiB in caches (293 GiB HF, 48 GiB uv), and 185 GiB in the homework
project. Pool had a separate 1,024-GB quota with only 99.7 GB used. The ordinary
`quota -s` output covered home; the authoritative report is `~/orcd/.quota`.

Engaging Tandem's stdout ended mid-configuration while scratch was full. Its
exit status is 1, but the missing traceback prevents attributing its exact final
exception. Compute-node local logs are inaccessible after the allocation ends.
Recovery scheduler logs now go directly to pool.

CSAIL Tandem reached step 123 and saved 120, then OOMed in backward: a 3.60-GiB
allocation failed with 3.27 GiB free on a 79.18-GiB GPU. The dynamic training
microbatch token budget is reduced from 10,000 to 4,096; batch 16, PPO minibatch
8, rollout count, token limits, learning rate and loss remain unchanged. Gradients
accumulate over smaller microbatches. Host RAM is raised from 144 to 160 GiB
because sampled RSS reached roughly 144 GiB during checkpointing.

CSAIL Solo and both Engaging pilot arms exhausted the old three-restart guard.
Recovery permits twelve bounded preemption/walltime restarts, with host-side
signals and complete-checkpoint checks on both container and native jobs.
The old attempt's unsaved steps are replayed, never promoted to a checkpoint.

Checkpoint storage fixes in the pinned verl patch:

- Retain HF weights through hard links on one filesystem, or staged copies
  across filesystems. Failure during staging preserves an existing snapshot.
- Persist separate HF snapshots only at reproduction validation steps (20) or
  the pilot's fixed final step (100). Resume state is still saved every 10 steps.
- Seed retention with the loaded checkpoint, preventing one orphaned optimizer
  snapshot per restart. Keep two actor checkpoints so failure writing later
  dataloader state or the commit marker cannot delete the last committed actor.

The reproduction campaign and existing pilot HF snapshots move to
`/orcd/pool/005/cge7/tandem-rlvr-recovery-20261008`, with checksum verification
before scratch copies are removed and symlinks preserving paths. Containers bind
both storage roots. Homework and other projects' models are preserved. The latest
CSAIL Tandem step 120 is transferred separately; Engaging Solo keeps step 120.

| Cluster | Training | Downstream jobs |
|---|---|---|
| CSAIL (cancelled duplicate chain) | Tandem 2596961; Solo 2596962 | Selection 2596963; four evals 2596964–2596967; plot 2596968 |
| Engaging reproduction | Gate 25248793; Tandem 25248795; Solo 25248796 | Selection 25248797; evals 25248799, 25248800, 25248802, 25248803; plot 25248804 |
| Engaging pilot | Array 25248946, budgets 2048/3072, saved steps 10/60 | Checks 25248947; evals 25248948; review 25248949 |

All training uses six-hour resumable allocations. No array throttle or dependency
serializes independent training. Engaging GPU work waits on successful recovery
verification. CSAIL jobs retain Nice 0 and the policy's vision-shared fallback;
Torralba GPUs were occupied by non-preemptible owner/interactive jobs, and neither
shared route offered an immediate start in submission dry runs. The completed
Engaging base evaluation is reused on both clusters. Earlier run notes below are
historical snapshots.

At 00:20 PDT the storage move and CPU import verification completed successfully.
The moved files were verified by rsync's whole-file transfer checksums and a
no-change metadata pass before source removal. A subsequent disk scan measured
776,765,394,944 bytes (723.4 GiB) in scratch and 485,672,526,848 bytes (452.3 GiB)
in pool; the site quota report lags by about 30 minutes. Engaging Tandem, Solo,
and pilot task 0 started on A100 80 GB GPUs; pilot task 1 remained pending on the
four-GPU preemptible limit. CSAIL Solo also resumed and completed step 121 with
finite loss/gradient metrics and the smaller microbatch; Tandem remained queued.
Startup success does not yet verify a new checkpoint write or completion of the
previously failing Tandem batch. Storage/retention, prior-checkpoint preservation,
cross-filesystem copy failure, history integrity and continuation checks pass.
At 00:25 PDT both Engaging reproduction logs confirmed loading model, optimizer,
RNG and scheduler from step 120. Pilot task 0 confirmed the same from step 10.

Storage planning: one full resume checkpoint is approximately 61 GiB and an HF
weight snapshot approximately 15 GiB. From step 120 to 200, retaining four new
validation snapshots per reproduction arm adds at most about 120 GiB; allowing
an additional full checkpoint per arm and both pilot final HF copies gives a
conservative pool growth allowance of roughly 275 GiB, excluding unrelated
projects and unusually large logs/evaluation outputs. Pilot checkpoint rotation
needs roughly 125 GiB of additional scratch headroom. These fit the post-move
scan's roughly 570/300 GiB pool/scratch margins, but the first new save is still
unverified. The site quota report observed at 00:32 PDT was stale (00:11 PDT).

The 4,096-token setting controls microbatch packing and log-probability scoring,
not the response length limit. The pinned FSDP engine computes the full
minibatch's valid-token denominator before splitting it into microbatches, sums
their gradients, then takes the optimizer step. Thus the intended token-mean
objective and effective batch are unchanged; throughput and floating-point
rounding can change, so the trajectory is not guaranteed bit-for-bit identical.

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

Both CSAIL training jobs were preempted again at 07:16 / 07:29 PDT, after logging
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

Engaging continuation submissions: import/CPU gate **25173823**, Tandem
**25173824**, Solo **25173825**, checkpoint selection **25173826**, independent
GRPO solo / Tandem solo / GRPO handoff / Tandem handoff evaluations
**25173827–25173830**, final CPU figure **25173831**. Campaign:
`/orcd/scratch/orcd/013/cge7/tandem-rlvr/figure2-resume-20261007`.
The direct transfer completed and CPU gate 25173823 passed at 09:59 PDT,
verifying source-history hashes, retained best weights, data hashes and saved
resume state. Solo 25173825 started on node2000 (A100) at 09:59 PDT; Tandem
25173824 remains queued. At 10:00 PDT Slurm forecast Tandem at 13:50 PDT today,
with `QOSMaxGRESPerUser` as its pending reason; this is a changing estimate,
not a reservation. The four preemptible GPUs were occupied by Solo, the running
pilot arm and two NLA jobs. By 10:03 PDT the forecast had moved to 10:08 PDT,
illustrating its instability. Solo's log confirms loading the step-120 model,
optimizer, RNG and learning-rate scheduler at 10:03 PDT; no new optimizer step
had completed at this check. Both CSAIL training jobs remain untouched.
