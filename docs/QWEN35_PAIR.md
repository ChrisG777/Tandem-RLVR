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
  and writes verified traces atomically. It does not establish training readiness.
  `slurm/check-pair.sbatch` wraps it with explicit site/environment arguments.
- `slurm/train-figure2.sbatch` uses `base` to initialize both training arms and
  the optional `junior` for the frozen partner; old base-only manifests retain
  self-pairing. A campaign `word-boundaries.json` selects its tokenizer boundaries
  and explicitly disables thinking during training.
- `eval/select_figure2.py` preserves both initial model records and verifies
  initialization identity before selecting the trained checkpoints.
- `slurm/figure2.sbatch` supports `junior-solo`, `base-solo`, and `base-handoff`
  in addition to existing phases. A paired manifest's full evaluation includes
  four solo and three handoff phases. `eval/figure2.py --paired-baselines`
  summarizes/plots the same seven outputs, preserving the original statistics.

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

Inference probe estimate: approximately 22 GiB model weights plus 10 GiB
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
