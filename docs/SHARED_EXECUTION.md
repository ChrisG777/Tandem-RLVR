# Shared RL execution

`train/execution.sh` exposes `configure_training_execution MODEL_FAMILY MODE`.
Supported families are `qwen3` and `qwen35`; modes are `baseline` and `optimized`.
It sets `EXECUTION_EXTRA`, a Bash argv array of verl/vLLM overrides, and execution
kernel/thread variables. It never chooses data, rewards, models, RL batch sizes,
sampling parameters, token budgets, checkpoints, or training horizon.

Qwen3.5 speed mode, APPS and elegance reuse this helper. APPS/elegance opt in with
`TRAIN_EXECUTION=optimized`. It reuses the existing patched verl/vLLM implementations:
deferred FSDP gradient synchronization, native fused AdamW, FLA cross-entropy,
chunked entropy and decode-only CUDA graphs. Qwen3.5 requires the qualified CUDA
toolkit and the captured-FA3-metadata fix. APPS retains its FSDP1 layout and qualified
32/64-sequence, 48/96-GiB KV settings; elegance retains padded FSDP2 CPU offload.
`TRAIN_PROFILE_DIR` optionally enables the existing per-stage diagnostic hook;
leave it unset for end-to-end timings because synchronization adds overhead.

`train/profile-experiment.sh apps|elegance grpo|tandem` runs three disposable
full-weight steps and full validation within an existing interactive allocation.
Set `PROFILE_ROOT` explicitly. Production artifacts are never inputs or outputs.
Both wrappers save raw rollouts. The elegance profiler has an isolated ephemeral
judge archive, sharing the production result cache and four-slot judging limit.

Qualification is pending: compare timed stages and generated tokens, check finite
updates and native authorship, inspect cap hits/raw outputs, require complete
validation, then separately validate checkpoint/resume before production rollout.
The current production jobs still use their original source checkouts.

Elegance's response limit is 6,000 tokens for both arms and profiling (increased
from 3,000 on October 12 at the user's request). The rollout context limit follows
the prompt plus response budgets. Continuations retain their earlier checkpoints;
the submission receipt records the step where the budget changed, so earlier
3,000-token results must not be treated as results under the new budget.
