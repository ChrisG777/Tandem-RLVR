# Qwen3.5 execution profiling

The slow 200-step campaigns were cancelled at the user's request; their saved
checkpoints remain intact. Target: approximately 24 GPU-hours per complete run,
including validation/checkpoint overhead, with the original objective, batches,
rollout count, response budget, and full-weight training retained.

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
four OpenMP threads, and decode-only graphs. Supply `PPO_MAX_TOKEN=16000`
explicitly; this changes execution microbatch packing, not the RL batch or
minibatch. Full-weight FP32 parameters/optimizer states and BF16 compute remain.
`MODE=smoke SMOKE_STEPS=3 SMOKE_VAL_SAMPLES=1000` runs three steps and the complete
held-out pass@4 validation. `QWEN35_PROFILE_DIR=DIR` optionally enables stage timing.
All other campaigns retain their defaults. These options require both updated
fork patches and the pinned Qwen3.5 runtime described above.

Restartable short jobs 2622947 (Tandem) and 2622948 (GRPO) started independently
on one A100 80 GB each. They use source commit `0c726c3`, full validation, and
separate output roots under `speed/fused-ce/{tandem,grpo}-smoke`. Each saves a
recovery/HF checkpoint at step 3, automatically resumes committed progress on
requeue, and stops after the short test. They do not restart the cancelled
200-step campaigns. A separate checkpoint-resume qualification is still needed.

The target remains **unproven**. For one GPU, estimate complete-run GPU-hours as
`(200 × steady_step_seconds + 10 × full_validation_seconds + checkpoint_seconds
+ startup_seconds) / 3600`, without double-counting validation/save time if the
reported step timer includes them. Keep validation every 20 steps and include
the actual recovery checkpoint cadence. The absolute step-only limit is 432 s;
validation and checkpoint overhead require a lower steady-state step time.
