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
This candidate is not yet validated for the full Qwen3.5 run.

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
