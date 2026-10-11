# Qwen3.5-9B / Qwen3.5-2B Figure 2 extension

This campaign tests an initially asymmetric pair using the existing native
Tandem backend. It is an extension of Figure 2, not an exact reproduction of
the paper's self-paired initialization. Both models use non-thinking mode.

## Interfaces and provenance

- `eval/qwen35-models.json` pins official senior (`base`) and frozen `junior`
  revisions. `eval/download_figure2.py --manifest MANIFEST --out base.json`
  downloads them and records immutable local snapshot paths.
- `train/assets/build_word_boundary_ids.py --model SNAPSHOT --out
  word-boundaries.json --write` regenerates the existing leading-space rule for
  this tokenizer. `--expected-count N` optionally verifies a pinned count.
- `train/check_pair.py --models base.json --boundaries word-boundaries.json
  --out pair-probe.json` exercises the original native backend on one GPU,
  checks identical vocabularies, exact boundaries and per-token authorship,
  and writes verified traces atomically. The probe also changes/restores one senior
  parameter through vLLM’s weight loader and verifies all junior parameters remain
  frozen and byte-identical. It does not establish training readiness.
  `slurm/check-pair.sbatch` wraps it with explicit site/environment arguments.
- `slurm/train-figure2.sbatch` uses `base` to initialize both training arms and
  the optional `junior` for the frozen partner; old base-only manifests retain
  self-pairing. A campaign `word-boundaries.json` selects its tokenizer boundaries
  and explicitly disables thinking during training. `MODE=smoke` defaults to three
  updates; `SMOKE_STEPS=4` continues the same smoke root for a focused checkpoint
  restoration check. Full training remains fixed at 200 steps.
  Set `EXPECTED_RESUME_STEP=3 RESUME_LOG=PATH` on that continuation to require
  separate pre/post-resume metric attempts plus model, optimizer, RNG and scheduler
  load evidence. The existing checkpoint verifier stores this receipt in `selected.json`.
- `eval/select_figure2.py` preserves both initial model records and verifies
  initialization identity before selecting the trained checkpoints.
  `--arm grpo` or `--arm tandem` verifies/selects one local arm, enabling
  cross-cluster evaluation without copying all training checkpoints. Omit it
  to retain the existing two-arm selection workflow.
- `slurm/figure2.sbatch` supports `junior-solo`, `base-solo`, and `base-handoff`
  in addition to existing phases. A paired manifest's full evaluation includes
  four solo and three handoff phases. `eval/figure2.py --paired-baselines`
  summarizes/plots the same seven outputs, preserving the original statistics.
  For a separate CPU plot job, use `slurm/figure2-postprocess.sbatch` with
  `PHASE=plot PAIRED_BASELINES=1`.

The separate native word evaluator accepts `--boundaries` for diagnostics.
Native-word benchmark evaluation is not part of this campaign's requested scope.

## Protocol

Use the existing DeepScaleR split, seed 42, 200 steps, batch 16, minibatch 8,
eight rollouts, learning rate 1e-6, and 3,000-token response budget. Save and
validate every 20 steps. Select the earliest maximum held-out pass@4 for each
arm under its own rollout protocol. Use the corrected shared math grader.

Solo evaluations: initial 9B, frozen 2B, selected GRPO and selected Tandem,
32 draws per problem. Paragraph handoffs: initial 9B, GRPO and Tandem paired
with the same 2B, eight chains per problem. Retain the existing 1,064 problems,
decoding, macro weighting and bootstrap uncertainty estimates. No 4B baseline.

## Compatibility gate and execution

Keep the original sampler and loss. Both pinned upstream libraries already
implement Qwen3.5. First test the native Tandem backend without architecture
changes; fix only demonstrated integration errors using upstream components.
Do not substitute an Agent Loop or paragraph-level training. Substantial
architectural changes require discussion with the user.

Inference probe estimate: approximately 22 GiB model weights plus 5 GiB
explicit cache budget, with remaining headroom on a 48-GiB-or-larger GPU;
64 GiB host RAM and four CPUs. This is not a training memory estimate.
Training and full evaluation remain gated on compatibility and focused smokes.
All cluster routes follow the shared allocation policy, including explicit RAM,
maximum route walltime, broad compatible hardware, requeue and no duplicates.

Implementation checkout: `Tandem-RLVR-qwen35`, branch `codex/qwen35-pair`.
The original dirty working tree is untouched; commit `a758779` captures its
starting code layout separately from experiment changes.

Initial CSAIL jobs: model preparation `2615764`; native backend probe `2615771`.
Artifacts: `/data/vision/torralba/u/chrisge/tandem-rlvr/qwen35-9b-2b`.
The probe uses the existing training environment read-only; no installed model
or fork code is modified by it.

## Paper context

The original paper's [Section 3.2](https://arxiv.org/html/2606.28166v1#S3.SS2)
explicitly allows asymmetric pairs, while choosing identical initial models
to isolate RL-induced changes from initial style/capability differences.
The revised [conclusion](https://arxiv.org/html/2606.28166v2#S6) proposes diverse
juniors as future work. This experiment measures the initial math gap itself;
model size is not treated as evidence of benchmark performance.

Probe 2615771 loaded both models successfully, then was cancelled during
FlashInfer's 35-target first-use kernel compilation. The diagnostic selects
vLLM's built-in Triton GDN prefill backend and disables unused image/video
profiling to avoid that startup cost; neither changes the Tandem architecture.

Native probe **2615854 passed**: four responses, 957 tokens, senior fraction
0.5350; every token had valid authorship and every switch passed the existing
word-boundary/backstop check. Probe 2615815 exposed the redundant junior cache
allocator selecting the hybrid backend's unsupported ordinary-attention API.
A 20-line addition delegates colocated hybrid cache allocation/metadata to
upstream vLLM. Probe 2615838 then exposed the optional prefix-cache copier's
identical-Mamba-shape assumption. Disabling prefix caching via supported config
resolved that failure. No model, kernel, switching or loss implementation changed.
The hybrid path currently requires both decoders on the same GPU.

Engaging setup **25536318 completed** using the existing setup entrypoint and
an isolated training environment with the pinned grading dependencies. The
pending CSAIL CPU-check job 2615841 was cancelled before this work moved.
Training uses text-only mode, frozen vision tower, Triton GDN prefill, and
disabled prefix caching. These are supported execution settings, not changes
to the RL objective. Training/checkpoint validation remains outstanding.

Three-step training smokes: CSAIL Tandem `2615877`, Engaging solo GRPO
`25536736`. Each requests one H200, six CPUs and 320 GiB host RAM; host RAM
is an estimate for model loading, CPU optimizer offload and checkpoint buffers.
The single-GPU native probe established colocated decoding support. Training
memory peaks and checkpoint/resume behavior must still be measured before full
launch. Walltimes are 24 hours on CSAIL shared and six hours on Engaging's
combined normal/preemptible route.

The CSAIL smoke composes the existing read-only training packages with the
isolated patched vLLM source and the existing corrected grading packages using
`PYTHONPATH`. Training site-packages precede grading site-packages to preserve
NumPy 2.2.6, pandas 3.0.4, torch 2.10.0 and transformers 5.5.4; math-verify 0.9.0
and latex2sympy2-extended 1.11.0 come from the grading environment. Engaging uses
a separately built environment recorded in `training-environment.txt`.

The first training smoke attempts were cancelled before accepting training
results: the pinned Transformers Qwen3.5 GDN forward has neither sequence
boundaries nor position IDs in its interface, so verl's packed/remove-padding
path carries recurrent state across examples. A two-sequence check using the
installed `torch_chunk_gated_delta_rule` produced a 1.39206 maximum absolute
difference on the second sequence when concatenated instead of batched. The
Qwen3.5 wrapper now sets `model.use_remove_padding=False`, retaining the existing
padded SDPA path and separate recurrent state per example. This changes execution
layout, not batch size, token budgets, or the objective.

Corrected training smoke jobs are CSAIL `2615902` and Engaging `25536909`;
the original attempts `2615877` and `25536736` were cancelled. Evaluation
smokes `2615910`–`2615912` failed because the isolated vLLM source lacked a
generated FlashAttention Python module used during vision profiling. Their
replacements `2615918`–`2615920` reuse the existing complete vLLM installation
read-only; solo and paragraph evaluation do not need the native hybrid cache
patch. Each smoke uses two problems and two samples/chains, separate from full
benchmark outputs.

Weight audit `2615929` passed: all junior weights remained frozen and
byte-identical through a senior parameter reload, and the senior parameter was
restored. The local diagnostic enables vLLM callable serialization for its own
worker audit. Non-thinking prompt text and token IDs also matched. The 2B solo
smoke `2615918` passed; full 32-draw/1,064-problem baseline is job `2615938`.
GRPO smoke `25536909` was preempted once and automatically requeued under the
same ID, before any checkpoint was written. This verifies scheduler recovery,
not checkpoint restoration.

The complete evaluator installation resolves the missing module. On H200,
9B solo/handoff smokes `2615919`/`2615920` then spent over six minutes compiling
FlashInfer GDN prefill variants, reproducing the earlier native-probe startup
cost. They were cancelled before generating samples. `eval.common.build_engine`
now selects built-in Triton GDN prefill and disables image/video profiling for
Qwen3.5 models only, matching the validated native execution settings. The 2B
RTX 3090 path already selected Triton automatically. Decoding is unchanged.

Tandem smoke `2615902` reached the first optimizer update, then exhausted the
139.8-GiB H200 while FSDP1 lazily allocated Adam's second moment. The observed
policy-process usage was 134.41 GiB plus 4.60 GiB for the frozen rollout partner.
The inherited manual optimizer offload does not avoid the on-GPU optimizer step.
The wrapper now selects verl's existing `actor.strategy=fsdp2` and
`actor.fsdp_config.offload_policy=True`, matching its pinned Qwen3.5 example.
Upstream `CPUOffloadPolicy` keeps parameters, gradients and optimizer state on
CPU between computations; no trainer or optimizer implementation is replaced.
The pending GRPO retry `25536909` was cancelled before applying this correction
to both arms. No completed training checkpoint exists yet.

FSDP2 retry jobs: Tandem `2615963` on CSAIL and GRPO `25537766` on Engaging.
The offloaded layout admits one 80-GB-class GPU or larger (an estimate pending
the smoke). CSAIL includes all A100/H100/H200 partitions. Engaging includes
verified A100-80GB, H100, H200 and RTX Pro 6000 nodes across both GPU routes;
smaller or unverified cards are excluded using the live inventory and ORCD's
[GPU memory table](https://orcd-docs.mit.edu/running-jobs/available-resources/).

Initial 9B solo smoke `2615943` passed. Paragraph handoff smoke `2615944` is
still running. Evaluation wrappers preserve per-job source commits and copies
of model manifests as `code-JOBID.txt` and `models-JOBID.json`, so independently
scheduled phases retain their own provenance.

Cross-cluster source trees match exactly: vLLM tree
`d44a294e9073207c653e3a38684c9377939fd299`, verl tree
`1fd0e2d493d6ca6f063f31e3fe06eccc7840b702`. Both environments use torch 2.10.0,
transformers 5.5.4, Triton 3.6.0, FlashInfer 0.6.6, Ray 2.55.1, TransferQueue
0.1.8, NumPy 2.2.6 and pandas 3.0.4. Full package inventories are saved in
the campaign directories. No optional FLA/causal-conv training kernels have
been added; Transformers currently uses its existing PyTorch training kernels.

All three evaluation smokes passed. Full baseline jobs are frozen 2B solo
`2615938`, initial 9B solo `2615969`, and initial 9B/2B paragraph handoff
`2615977`. Handoff requests one 80-GB-class-or-larger GPU, four CPUs and 32 GiB
host RAM, based on a sampled 19-GiB smoke RSS peak with headroom. The 9B engine
required about 20.6 GiB before its KV cache; at the existing 0.42-per-engine
memory fraction, 48-GB cards do not have sufficient cache headroom.

Optional accelerated training kernels are pinned in
`env/qwen35-training-kernels.txt`. Install into a separate directory with
`uv pip install --target "$RUN_ROOT/training-kernels" --python-version 3.11
--only-binary :all: --no-deps -r env/qwen35-training-kernels.txt`, then prepend
that directory to training's `PYTHONPATH`. Existing torch/Transformers/Triton,
einops, apache-tvm-ffi and torch-c-dlpack-ext satisfy the remaining dependencies.
Set `CUDA_HOME` to the site's CUDA toolkit for FLA's native TileLang backend.
Per-job `runtime-JOBID.json` records the effective package versions and paths.

FLA 0.5.2 rejected its Triton backward path on Hopper with the pinned Triton 3.6
because of a known correctness bug. Its maintained TileLang fallback preserves
the existing core package pins. TileLang 0.1.9 was incompatible with installed
TVM FFI 0.1.12; TileLang 0.1.15 supports it. The GPU check
`tests/test_qwen35_kernels.py` passed with relative RMS errors 0.52–0.68% for
output and all five input gradients against Transformers' reference, using
two independent BF16 sequences and a partial final chunk. The fallback-only
smokes are being replaced before any accepted update/checkpoint; inference
evaluations continue unchanged.

Accelerated three-step smokes: CSAIL Tandem `2616100`, Engaging GRPO `25539689`.
Their strict fourth-step resume checks are `2616101` and `25539690`, respectively,
each dependent on smoke success. Fallback attempts `2615963`/`25537766` and their
pending continuations `2616017`/`25538683` were cancelled before replacement.
No full 200-step training arm has launched yet.
