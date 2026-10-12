# Qwen3.5 Hopper rollout investigation

The optimized Tandem Hopper runs exhibited incoherent, repetitive generations,
including before the first gradient update. This is not evidence about the
effectiveness of Tandem with unequal model capabilities.

## Evidence and controlled comparison

- Healthy A100 smoke `2623385`: source `733cc7b`, training rewards
  46.1%, 43.0%, 35.2% at steps 1–3.
- H100 resume `2624813`: source `1de8bae`, rewards 4.7% and 0.8% at
  steps 4–5; 93.8% and 97.7% of responses reached 3,000 tokens.
- Fresh H200 Tandem `2624814`: zero reward at steps 1–3 and 5;
  therefore checkpoint resume and accumulated gradient updates are not
  necessary causes. Cancelled after inspecting the collapse.
- H200 GRPO `2624812` did not exhibit the collapse and was left running.

The effective A100/H100 Hydra configs differed only in diagnostic entropy
compilation, total steps, validation/save cadence, and experiment labels.
There was no vLLM/Tandem patch change between these two source revisions.
Model revisions, word boundaries, decoding, and training hyperparameters match.

Read-only retrieval from the live TransferQueue preserved current H200 step-7
responses in `optimized-200/tandem-full/live-rollouts-2624814.json` and
`live-rollouts-long-2624814.json` under the campaign root. The latter contains
24 length-limited samples across six prompts, intentionally selected for length;
it is not an unbiased accuracy sample. Responses often begin with relevant
reasoning, then drift into repeated digits and unrelated text. Short arithmetic
responses can finish correctly. Prompt formatting includes the expected empty
thinking block.

The campaign root is
`/data/vision/torralba/u/chrisge/tandem-rlvr/qwen35-9b-2b` on CSAIL.

## Diagnostic interface

[`train/check_pair.py`](../train/check_pair.py) reuses the native engine,
authorship checks, frozen-weight fingerprint, and existing math grader.
`--long-rollouts` runs four fixed arithmetic problems with four samples each,
temperature 0.6, top-p 1, top-k -1, and a 3,000-token response cap. It saves raw
text/token/authorship records before grading. These small diagnostic problems
are separate from benchmark evaluation and are not a performance estimate.

- `--cuda-graphs`: decode-only graphs, with compiler mode 0.
- `--flash-attn-version {2,3}`: upstream attention implementation selection.
- `--isolate-attention-metadata`: diagnostic worker override that disables
  cross-group metadata reuse, forcing each builder to refresh its own buffers.

The four-way comparison is recorded under `collapse-diagnostic/`:
FlashAttention 3 + graphs, FlashAttention 3 eager, FlashAttention 2 + graphs,
and FlashAttention 3 + graphs with metadata reuse disabled.

An initial diagnostic attempt failed in the grader because its Python path
prioritized an incomplete regrading SymPy installation. Its outputs were not
saved. The corrected environment prioritizes the training environment, matching
production, and passed a boxed-answer grading preflight. Array `2625088` runs
the corrected comparison; raw outputs are now committed before grading.

## Confirmed defect and fix

The controlled comparison produced:

| H200 execution | Correct / 16 | Reached 3,000 tokens |
| --- | ---: | ---: |
| FlashAttention 3 + decode graphs | 1 | 15 |
| FlashAttention 3 eager | 16 | 0 |
| FlashAttention 2 + decode graphs | 16 | 0 |
| FlashAttention 3 + decode graphs, metadata isolated | 16 | 1 |

The isolated path's one length-limited response retained the correct boxed
answer and repeated relevant reasoning; this is distinct from the widespread
incoherent collapse. These small diagnostics establish a regression and its
remedy, not general model accuracy.

The defect is stale per-builder FlashAttention 3 scheduling buffers:
graph capture records a builder's own buffer address, whereas cross-group
`update_block_table` can return metadata from another builder without refreshing
that captured buffer. The fix disables metadata reuse only for FlashAttention
builders using both full CUDA graphs and ahead-of-time scheduling. The existing
upstream `build()` implementation then refreshes each group's persistent buffer.
Graphs, FlashAttention 3, the native sampler, and hybrid state management remain
enabled. No model, optimizer, objective, or decoding settings change.

All four array tasks `2625088_0`–`2625088_3` completed successfully. The two
existing hybrid-cache delegation tests pass; their source extractor was fixed
to stop at the next file's patch rather than accidentally parsing later hunks.
Both full patches passed `git apply --check` against their pinned library bases
and were applied in the isolated `/data/scratch/chrisge/Tandem-RLVR-qwen35-fixed`
checkout at source revision `59049bd`. Compiled extension binaries are reused
unchanged; running GRPO's source checkout is untouched.

Replacement job **2625164**, `q35-tandem-200-fixed`, was submitted to the
compatible vision-shared H100/H200 route with one GPU, eight CPUs, 384 GiB host
RAM, 24-hour walltime, requeue, and bounded automatic resume. Its launch script
is `optimized-200-fixed/tandem-full-submission.sh`. It starts from the pinned
base, not any collapsed-run checkpoint. Before the first training launch it
runs the long probe against the actual patched library (no diagnostic override),
requiring at least 14/16 correct and at most two length-limited responses.
The atomic diagnostic result is reused after requeue. It then runs the existing
200-step protocol with validation/checkpoint selection every 20 steps and
latest-only optimizer retention. At submission it was pending resources; the
actual patched-runtime preflight and new training rewards were not yet observed.

Replacement Qwen speed runs enable verl's existing `trainer.rollout_data_dir`
and `trainer.validation_data_dir`, preserving text, scores, and ground truth in
the checkpoint directory. The healthy GRPO run is not hot-patched.
