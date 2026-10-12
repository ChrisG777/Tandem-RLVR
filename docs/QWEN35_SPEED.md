# Qwen3.5 execution profiling

The slow 200-step campaigns were cancelled at the user's request; their saved
checkpoints remain intact. Target: approximately 24 GPU-hours per complete run,
including validation/checkpoint overhead, with the original objective, batches,
rollout count, response budget, and full-weight training retained.

## Current launch and retention policy (October 11, Pacific evening)

H100 retries `2624570`/`2624571` failed on their first backwards, before a new
training step was recorded. FLA 0.5.2 rejects Triton 3.6's known Hopper gated
backward correctness issue. TileLang was installed, but the configured system
CUDA toolkit was absent on these nodes. The portable NVIDIA compiler/headers
are now installed at `qwen35-9b-2b/cuda-12.8.1` using
[`env/install-qwen35-cuda.sh`](../env/install-qwen35-cuda.sh), with SHA-256-pinned
archives from NVIDIA's CUDA 12.8.1 redistribution manifest. Python/library pins
remain unchanged. Set `CUDA_HOME` to that directory; the Qwen speed wrapper
rejects missing `nvcc` before loading models. FLA automatically selects its
existing TileLang backend on affected Hopper devices.

New resume checks are GRPO `2624811` and Tandem `2624813`. Each first runs the
small forward/backward parity tests, then resumes steps 4–6 and full validation.
Successful load/metric/validation verification releases only that arm's fresh
200-step campaign: GRPO `2624812`, Tandem `2624814`. Full outputs are under
`qwen35-9b-2b/optimized-200/{grpo,tandem}-full`. The full runs start from the pinned
base models, preserving the original 20-step checkpoint-selection schedule.
Both arms request one H100/H200, eight CPUs, 384 GiB RAM, 24 hours, automatic
requeue/resume, and normal priority. Invalid dependencies cancel their successors;
there are no duplicate full-run continuations. Full jobs disable the diagnostic
per-microbatch synchronization hook and save every 20 steps.

The patched trainer now calls `commit_latest_checkpoint(root, step)` after every
successful synchronous save. It atomically replaces the latest-step marker and
then removes all other local actor/critic optimizer shards, including abandoned
partial saves. Model and HF weights remain available for evaluation/selection.
Two optimizer states can briefly coexist while the new checkpoint is being
written; deleting the previous state earlier would make a failed save destructive.
After commit, only the latest checkpoint retains optimizer state. Five storage
tests cover failed commit, idempotent cleanup, multiple ranks, weight retention,
cross-filesystem HF persistence and resumed checkpoint history. Async checkpoint
writing is not used by these jobs.

The three kernel tests passed on both H100 nodes in 82.5/85.0 seconds. Gated
DeltaNet output/gradient relative RMS errors versus the Transformers reference
were 0.52–0.68%; fused cross-entropy log-probability max error was 9.54e-7.
Full-model initialization/resume is underway. No completed H100 step or
24 GPU-hour complete-run result is claimed yet. Cluster execution is pinned to
`1de8bae`; the subsequent optional-critic retention guard is irrelevant to these
critic-free GRPO jobs and has not been hot-applied to their running checkout.

Storage cleanup authorized October 11: obsolete checkpoints from the cancelled
Qwen campaigns and old reproduction attempts, plus old matrix/concept/length
pilot artifacts, were removed on both clusters. Estimated released file allocation
after accounting for hard links: 988.38 GiB on CSAIL and 833.41 GiB on Engaging.
CSAIL reported approximately 1.3 TiB free afterward. The active optimized step-3
checkpoints are preserved, as are the original reproduction's selected HF models
(GRPO 160, Tandem 180) and both step-200 final checkpoints. Retained HF shard
length/header verification passed. Historical non-final checkpoint paths in older
notes below may therefore no longer exist. Logs/results outside retired pilot
directories, model caches, environments, and current unrelated campaigns remain.
Deletion receipts list exact paths, inode identities and completion status:
`/data/vision/torralba/u/chrisge/tandem-rlvr/cleanup-20261011.json` and
`/orcd/pool/005/cge7/tandem-rlvr-recovery-20261008/cleanup-20261011.json`.

Status, October 11 evening (Pacific): GRPO three-step qualification `2623390`
completed successfully. Tandem `2623385` completed all three updates and strict
full validation, but its final file-logger flush failed with `EDQUOT`; subsequent
artifact verification therefore failed. Its exact step-3 numeric record was
recovered from the complete console line into
`metrics-2623385-recovered-3.jsonl`, with source SHA-256 and method recorded in
`metrics-recovery.json`. No values were imputed. Existing checkpoint verification
then passed and wrote `selected.json`.

The H100 GRPO continuation `2623566` loaded model, optimizer, RNG, and scheduler
states but failed before logging a new step. Its shared log stopped while the
quota was exhausted, so the terminal cause is not established. Blocked Tandem
continuation `2623567` was cancelled. The rejected default-graph candidate's
remaining 17-GiB HF snapshot was deleted; valid checkpoints and all diagnostic
logs were preserved. Subsequent shared-volume free space was 143 GiB (this change
also reflects other activity; it was not all freed by that deletion).

Replacement resume tests `2624570` (GRPO, `andreas-h100-1`) and `2624571`
(Tandem, `isola-h100-1`) are running independently, one H100 each, from the valid
step-3 checkpoints. They retain the existing steps 4–6/full-validation scope,
automatic resume, 24-hour walltime, and no additional large checkpoint writes.
The old 200-step campaigns remain cancelled; these retries do not restart them.

Completed A100 warm steps 2/3 took 424.53/418.82 s for GRPO and
601.65/640.57 s for Tandem, subtracting the separate checkpoint save from step 3.
The full eligible pass@4 validation took 2,027.85 s (33.8 min) for GRPO and
3,490.92 s (58.2 min) for Tandem. Saves took 354.93/360.17 s. Using mean warm
step time, ten validations and ten saves gives about **30.0/45.2 GPU-hours**
for a 200-step GRPO/Tandem run on these A100 nodes, before startup. This is an
extrapolation from two steps and one validation, not a completed-run measurement.
There is still no completed post-optimization H100 step measurement.

Latest completed warm steps (short qualifications still running):

| Arm | Hardware | Step | Generation | Old-policy probabilities | Actor update | Training-only 200-step extrapolation |
| --- | --- | --- | --- | --- | --- | --- |
| GRPO | 1 A100 80 GB | 424.53 s | 68.70 s | 101.66 s | 248.02 s | 23.59 GPU-hours |
| Tandem | 1 A100 80 GB | 601.65 s | 118.77 s | 96.17 s | 379.45 s | 33.43 GPU-hours |

Neither number includes full-run validation/checkpoint overhead. These are step-2
measurements, not completed-run results or controlled comparisons with the old
campaigns on different hardware. The complete-run 24 GPU-hour target remains
unproven. One-H100/H200 resume qualifications are queued as described below.

`train/profile_qwen35.py` is an opt-in diagnostic import for the existing trainer:
set `QWEN35_PROFILE_DIR=DIR` and
`actor_rollout_ref.model.external_lib=train.profile_qwen35`. It synchronizes and
times existing forward/backward/optimizer calls, saving JSONL. Optionally set
`QWEN35_PROFILE_TRACE=1` for a bounded PyTorch trace of the second backward;
trace processing adds substantial diagnostic overhead. It does not implement a training loop.

Candidate optimization: opt-in backport of upstream verl's
`_gradient_sync_context` from commit
`1d367a4f459a43768f76312c0d3430a99ec46b8c`, using FSDP2's native
`set_requires_gradient_sync`. Set
`+actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=True`.
The default remains false to preserve existing runs. Gradient accumulation then
keeps gradients on GPU until the final microbatch, trading memory for fewer
CPU transfers. `tests/test_fsdp2_offload_accumulation.py` checks gradients and
AdamW updates against sync-every-microbatch on a small full-weight model.
The small gradient/update parity check and individual full-model steps passed;
complete training/validation/resume qualification is still pending.

## Interactive measurements (2026-10-11)

Allocation `2620920`: one H200 on `tedrake-h200-1`, eight logical CPUs and
384 GiB host RAM. Artifacts are under
`/data/vision/torralba/u/chrisge/tandem-rlvr/qwen35-9b-2b/speed/`.
The cancelled full jobs were CSAIL Tandem `2618075` and Engaging GRPO `25556548`.

The original Tandem step 1 spent 2,318 s in policy updates, 249 s in generation,
and 149 s in old-policy log probabilities (2,733 s total). The initial diagnostic
`micro10k-t4` restores the launcher's 10,000-token execution microbatch budget
and uses four OpenMP threads. Batch 16, minibatch 8, eight responses per prompt,
3,000 response tokens, learning rate, seed and full-weight objective are unchanged.

After warm-up, forward-only calls took about 2 s. The second backward's bounded
trace measured 50.37 s, including 37.58 s in device-to-pageable-host transfers
and 4.45 s of CPU addition. PyTorch 2.10 FSDP2 explicitly disables asynchronous
gradient offload when accumulating into an existing CPU gradient. This explains
the repeated synchronous copies that the upstream deferred-sync option avoids.
The diagnostic was stopped after obtaining this trace; it did not complete an
RL step and is not an end-to-end speed benchmark. Trace processing itself was slow.

The small CUDA parity check passed with exactly equal gradients and AdamW
updates. Full-model candidate `nosync10k-t4` uses the isolated patched verl
checkout, the same execution settings, and deferred synchronization enabled.

`nosync10k-t4` completed step 1 in 667.2 s: generation 112.6 s,
old-policy log probabilities 65.2 s, actor update 467.4 s, weight transfer 4.2 s.
Normal backward microbatches took about 3–4 s; compilation inflated several
calls to 15–41 s. Its two CPU AdamW updates took 81.7 and 88.7 s. Peak policy
allocation was 60.5 GiB (76.7 GiB reserved). These are cold-start diagnostic
numbers, not a 24-hour result. This comparison was stopped after step 1.

The same small parity test also passed with native PyTorch fused CPU AdamW:
exactly equal gradients and updated weights. The next candidate,
`tandem-fused16k-graphs`, uses the unchanged Tandem launcher with
`PPO_MAX_TOKEN=16000`, `OMP_NUM_THREADS=4`, `ENFORCE_EAGER=False`, deferred
synchronization and
`actor_rollout_ref.actor.optim.override_optimizer_config={fused:True}`.
No diagnostic hook is loaded. It requests three training steps followed by
the full 1,000-problem pass@4 validation and a recovery/HF checkpoint.

`train/check_pair.py --cuda-graphs` exercises the same native pair probe with
vLLM decode-only graph capture and torch.compile disabled, including per-token switch checks and a full junior
weight fingerprint across a senior weight reload. Use a separate `--out` path
from the eager probe; graph mode is part of its saved identity. The underlying
`experiments.word_handoff.word_handoff.build_engine(..., enforce_eager=True)`
keeps eager execution as the default for existing callers.

The default graph candidate is **rejected**: all three training batches hit the
response limit with zero rewards, and validation crashed with
`scheduler_metadata must have shape (metadata_size)` in the junior's FA3 call.
Its partial validation aggregates are invalid. The source diagnosis found
model-wide head counts in the pinned FlashAttention metadata builder, despite
the senior using 16 query / 4 KV heads and junior 8 / 2. An isolated backport
from upstream vLLM `f26d9f432ced7b1087d02d29bffed2249df1a18e` reads the actual
layer's query-head count and cache specification's KV heads/head size, and groups
metadata by query-head count. This is upstream metadata construction, not a
replacement sampler or cache manager. It still needs full-model validation.

The Qwen wrapper also sets `trainer.require_complete_validation=True`. This
small guard rejects missing or unexpected prompt/sample sessions before computing
validation aggregates. The pinned replay buffer otherwise accepts failed prompts
and can report partial pass@4 scores after an inference-engine failure.

The eager control `tandem-fused16k-eager` completed a valid first step in 562.97 s:
generation 198.75 s, old-policy log probabilities 73.43 s, policy update 271.09 s.
Reward was 0.4453, gradient norm 0.4667, and senior token fraction 0.4955.
This is a cold step on one H200, not a complete-run forecast.

The corrected `pair-decode-graphs` probe passed with actual head groups (16, 4)
and (8, 2), valid word switches, and identical junior weight fingerprints before
and after senior weight reload. Decode-only graphs disable torch.compile using
`compilation_config={mode:0,cudagraph_mode:FULL_DECODE_ONLY}`.
The subsequent `tandem-decode16k` training attempt was terminated during startup
when interactive allocation 2620920 ended; it produced no step metrics.

Profiling also found thousands of per-token log-softmax/gather backward calls in
verl's fallback log-probability calculation. The already installed FLA 0.5.2
provides the same cross-entropy interface used by verl's FlashAttention path.
`VERL_USE_FLA_CROSS_ENTROPY=1` selects that existing kernel explicitly and fails
on import if it is unavailable. No kernel or loss was reimplemented. The
248,320-token vocabulary parity check passed for ordinary and in-place backward:
maximum log-probability error 9.54e-7 and gradient relative RMS error 7.66e-6
against FP32 PyTorch log-softmax. Full-model speed and memory remain to be measured.

## Reproducible short qualification

The existing `slurm/train-figure2.sbatch` accepts `QWEN35_SPEED=1` to enable the
candidate deferred synchronization, native fused CPU AdamW, FLA cross entropy,
512-token chunked entropy, four OpenMP threads, and decode-only graphs. Supply `PPO_MAX_TOKEN=10000`
explicitly; this changes execution microbatch packing, not the RL batch or
minibatch. Full-weight FP32 parameters/optimizer states and BF16 compute remain.
`MODE=smoke SMOKE_STEPS=3 SMOKE_VAL_SAMPLES=1000` runs three steps and the complete
held-out pass@4 validation. `QWEN35_PROFILE_DIR=DIR` optionally enables stage timing.
The source held-out split has 1,000 problems; the unchanged 1,024-token prompt
filter retains 999 with this tokenizer, so complete validation means 3,996
sessions. This training filter does not change the separate 1,064-problem benchmark.
All other campaigns retain their defaults. These options require both updated
fork patches and the pinned Qwen3.5 runtime described above.

Short jobs 2622947 (Tandem) and 2622948 (GRPO), source `0c726c3`, both failed
before the first update on A100 80 GB. The padded forward path bypassed verl's
configured chunked entropy function and allocated another 21–23 GiB while
logging old-policy entropy. A six-line integration now calls the existing
chunked function on flattened padded logits and restores the output shape.
The value/gradient check against unchunked entropy passed. This does not remove
entropy logging or change the objective, and all failure logs are retained.

Replacement restartable short jobs 2623089 (Tandem) and 2623091 (GRPO) started
independently on one A100 80 GB each, source `6b4867f`, with full validation and
separate output roots under `speed/fused-ce/{tandem,grpo}-smoke`. Each saves a
recovery/HF checkpoint at step 3, automatically resumes committed progress on
requeue, and stops after the short test. They do not restart the cancelled
200-step campaigns. A separate checkpoint-resume qualification is still needed.

The entropy fix reduced measured forward-only peak allocation from over 71 GiB
to 17–19 GiB, with warm calls around 3.6–4.1 s. Both 16,000-token candidates
then failed during accumulated-gradient backward: PyTorch's padded nested-tensor
backward needed another 11.7–12.7 GiB alongside the resident FP32 gradients.
The 16,000-token setting is therefore **not qualified for 80 GB GPUs**.
The next attempts use the previously measured 10,000-token execution budget;
the RL batch, minibatch and accumulated update are unchanged. Both failed
attempts ended without a checkpoint. All logs remain in their original files.
Current short attempts are Tandem `2623239` and GRPO `2623255`, source `6b4867f`,
one A100 80 GB each. They retain the same bounded three-step/full-validation
protocol and output roots, with distinct per-attempt logs and runtime manifests.

Both 10,000-token attempts completed the first policy update but failed while
waking vLLM weights, before a full step could be logged. Warm backwards were
about 4–6 s with peak policy allocation below 59 GiB; Tandem's first fused CPU
optimizer call took 37.1 s with gradient norm 0.5125. The pinned weight-update
path retained unused allocator memory before waking the separate rollout process.
Upstream verl `1d367a4f459a43768f76312c0d3430a99ec46b8c` already adds
`get_torch_device().empty_cache()` at that exact boundary. This one-line fix is
backported in `733cc7b`; the profiler now records reserved as well as allocated
GPU memory. No model/state-management replacement is involved. Retry Tandem
`2623385` and GRPO `2623390` use this source. Both are running on A100 80 GB
(`beery-a100-2` and `beery-a100-1`, respectively). The failed GRPO process was
cancelled during shutdown after recording its fatal wakeup error.
No completed-step timing exists for the failed attempts.
The retries reused unchanged node-local compiled-kernel caches where available;
their first-step timings should not be described as wholly cold starts.

GRPO `2623390` completed a valid first step in 513.44 s on A100: generation
76.68 s, old-policy probabilities 122.30 s, actor update 288.37 s, and weight
update/wake-up 5.69 s. Batch reward was 0.5547, gradient norm 0.4036, mean response
length 1,830 tokens, and no responses aborted. Peak policy allocation was
56.86 GiB (63.42 GiB reserved). This verifies the cache-release transition for
GRPO but is not a completed smoke or a 24 GPU-hour result.

Subsequent speed-mode launches also set
`actor_rollout_ref.actor.fsdp_config.use_torch_compile=False`. In this pinned
engine that flag controls only the diagnostic entropy function. Variable padded
shapes hit TorchDynamo's eight-recompilation limit in the first batch, then fell
back to the same uncompiled chunked function anyway. Disabling that redundant
compilation avoids its startup cost; it does not disable FLA's Triton kernels or
vLLM decode graphs. Already-running jobs retain their captured configuration.

GRPO's warm second A100 step took 424.53 s: generation 68.70 s, old-policy
probabilities 101.66 s, actor update 248.02 s, and weight update 5.86 s. Its
batch reward was 0.6406 and gradient norm 0.6844. At that rate, 200 steps alone
cost 23.59 GPU-hours, before required validation, saves, and startup. A100 is
therefore not qualified for the complete-run 24 GPU-hour target at this pace.

Single-H100/H200 resume profiles are queued behind each successful three-step
qualification: GRPO `2623566` after `2623390`, Tandem `2623567` after `2623385`.
They resume the complete step-3 checkpoint, run steps 4–6 with the same RL
settings and 10,000-token microbatches, then perform full eligible validation.
They use the existing training launchers directly, disable entropy compilation,
and do not save another large checkpoint or overwrite the step-3 selection.
Their exact Slurm scripts are saved as `resume-submission-JOBID.sh` in each
smoke root. Each requests one compatible Hopper GPU, eight CPUs, 384 GiB RAM,
24 hours, and requeue; restart loads step 3 again and retries are bounded at 12.
These are performance/recovery tests, not resumed 200-step campaigns. Final
`verify_resume` receipts and complete-run forecasts are still pending.

Tandem's second A100 step completed with reward 0.4297, gradient norm 0.4516,
senior token fraction 0.4989, and no aborted responses. Its two optimizer calls
took 62.54 and 84.04 s, versus roughly 23–24 s for warm GRPO calls. Both nodes
have AMD EPYC 7773X CPUs. Read-only affinity inspection found Tandem's worker
allowed CPUs `0,121-127` (one GPU-local core), versus GRPO's `56-58,123-127`
(three GPU-local cores); both workers' mapped host pages overwhelmingly resided
on GPU-local NUMA node 0. This suggests a CPU/memory placement contribution,
but it is not a controlled causal test. No live affinity changes were applied.
GPU model alone is therefore insufficient to predict offloaded optimizer speed.

The target remains **unproven**. For one GPU, estimate complete-run GPU-hours as
`(200 × steady_step_seconds + 10 × full_validation_seconds + checkpoint_seconds
+ startup_seconds) / 3600`, without double-counting validation/save time if the
reported step timer includes them. Keep validation every 20 steps and include
the actual recovery checkpoint cadence. The absolute step-only limit is 432 s;
validation and checkpoint overhead require a lower steady-state step time.

## Completed GRPO verification after preemption (2026-10-12)

The 200-step GRPO run saved final weights and validation, but its completion
check found missing step-160 metrics. The dataloader checkpoint has snapshot
step 160 and 160 yielded batches; intact HF weights at 160 and adjacent metric
attempts ending at 159 and starting at 161 remain. Slurm had overwritten the
startup log on requeue, so full-state loading at 160 cannot be independently
verified from that log.

[`record_saved_checkpoint_gap(root, step)`](../train/figure2_checkpoint.py)
(`--record-saved-step`) records this narrower evidence: exact saved dataloader
snapshot, its SHA-256, intact HF weights, and adjacent attempt boundaries.
It certifies the saved update, not optimizer/RNG loading. Missing metrics and
validation remain unavailable and are excluded from checkpoint selection.
Existing log-based resume qualification still requires all four load messages.
New training submissions append stdout across requeues to retain that evidence.
