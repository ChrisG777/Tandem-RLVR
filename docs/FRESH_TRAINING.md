# Independent Figure 2 training

Both policies start from official `Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. The frozen Tandem junior is the same
base snapshot. No authors' RL-trained checkpoint is used for initialization,
checkpoint selection, or evaluation. Earlier released-model evaluation jobs
2559028, 2559029 and their monitor 2559040 were cancelled on 2026-10-05.

The editable vLLM installation reports `0.19.2.dev1` because the patch adds one
commit after the `v0.19.1` tag. Its source is still the pinned
`b1388b1fbf5aaef47937fabe98931211684666a6` plus this repository's patch; native
extensions come from the matching 0.19.1 wheel. The resolved environment is
recorded in `RUN_ROOT/training-environment.txt`.

## Protocol

Reuse `train/vanilla_grpo.sh` and `train/tandem_grpo.sh` and the repository's pinned
vLLM/verl patches. Both arms run 200 optimizer steps, matching the horizon shown
in Figure 4. Validate and save every 20 steps. Select the highest held-out pass@4
under each arm's own rollout protocol (earliest step breaks ties), rather than
forcing the authors' selected steps 200 and 120. Test benchmarks never select
checkpoints. This is a single-seed reproduction, not a seed-averaged result.

Use the existing DeepScaleR preparation with 40,309 nonempty-answer examples,
1,000 held out with split seed 0, and 39,309 for training. Resolve the dataset's
Hub commit before loading; record it and the two parquet SHA-256 hashes.
Data-order and rollout seeds are 42 for both arms. Existing launcher defaults
retain batch 16, mini-batch 8, group 8, LR 1e-6, clipping 0.2, response 3,000,
training temperature 0.6, top-p 1, top-k -1, no entropy/KL penalty. Tandem uses
word handoffs, senior probability 0.5, gap cap 32 and zero junior-token loss.

GRPO reserves one GPU; Tandem reserves two, one for training and one for its
frozen junior. Both request six CPU cores and 144 GiB host RAM. Training accepts
A100 80 GB, H100 and H200 nodes with explicit memory-class feature constraints.
The CPU allocation is supplied explicitly to Ray. Full jobs have a 24-hour
limit; the paper reports 7.8 h to GRPO step 200 and 9.4 h to Tandem step 120 on
two A100 80GBs. Our Tandem three-step smoke took 39m12s on two A100 80GBs,
including startup, validation and checkpoint saving. Its first two steps took
10.4 and 8.1 minutes; extrapolating these short measurements suggests a full run
may need checkpoint resumption beyond one 24-hour allocation. Our single-GPU GRPO
layout differs from the paper's two-GPU layout while retaining its global batch,
mini-batch and optimizer settings. At most three scheduler
restarts are allowed, resuming only that run's own optimizer checkpoint.

## Resource budget (estimates, not measured minima)

The base safetensors index reports 8,045,591,552 bytes of BF16 weights, about
7.49 GiB. Its trainable FP32 parameters occupy about 15 GiB; two Adam moment
buffers add 30 GiB, and gradients another 15 GiB. GPU activation, mixed-precision
and kernel buffers add to those states during training. The paper's one-GPU
Tandem senior fits an A100 80 GB; 40/48 GB cards lack comfortable room for the
unsharded policy state with this configuration. Both training arms therefore
accept all observed 80 GB-or-larger A100/H100/H200 types, rather than H100/H200
only. Tandem's second GPU is a backend placement requirement, not a claim that
every implementation mathematically needs two GPUs.

The original 128 GiB host budget allowed roughly 45 GiB for offloaded parameters/Adam,
45 GiB for temporary checkpoint/loading copies, a capped 4 GiB Ray object store,
and 34 GiB for Python workers, data and transient headroom. Not all buffers are
live simultaneously. Completed Tandem smoke 2560589 reported peak batch RSS
134,207,636 KiB (128.0 GiB), leaving essentially no margin. Requests now use
144 GiB, adding 16 GiB (12.5%) above that observed peak. This is justified
headroom, not a measured minimum. The trainer's CPU-memory metric is not used
as a per-job peak measurement.

GRPO smoke 2560586 failed before training: vLLM's 0.8 memory fraction requested
63.34 GiB with only 62.45 GiB free beside FSDP. The batch environment now uses
0.65 for both arms (already the successful Tandem setting). This changes cache
capacity, preserving batch sizes, decoding and optimizer settings. Standalone
vanilla launcher defaults remain unchanged.

Each arm has one trainable-GPU placement group reserving three Ray CPUs. One
TransferQueue storage actor and its controller reserve one CPU each; a sixth
allows other actors to start and work to progress. The previous reduction to
eight CPUs overlooked TransferQueue's default **eight** storage actors and would
not have fit the total reservation. The campaign now sets one storage unit, two
DataLoader workers, four asynchronous rollout workers, and one OpenMP thread per
process. This is sufficient for the small single-node batches without changing
the learning objective, global batch 16, mini-batch 8 or rollout group size 8.

Evaluation uses its existing single-GPU engine-colocation mode, automatically
selected when only one GPU is visible. Each of the two 7.49 GiB models gets 42%
of VRAM, leaving room for KV cache on 40/48 GB-or-larger GPUs. The host estimate
is 48 GiB for two model-loading footprints, worker processes, generated token
objects and serialization buffers; four CPUs serve the engines and grading.
The previous 128 GiB inference request was unnecessarily conservative. No extra
GPU jobs were submitted merely to profile these resource estimates.

## Setup and interfaces

Clone the pinned upstream forks via SSH on CSAIL, then use
`bash third_party/apply_patches.sh` without `--install`.
`slurm/training-setup.sh` runs on a CPU allocation, creates a separate uv training
environment, installs the pinned dependencies and editable Python patches over
the matching vLLM binary wheel, runs tests, downloads only the base and builds
the common data split. Site inputs: `REPO`, `TANDEM_ENV`, `HF_HOME`,
`UV_CACHE_DIR`, `RUN_ROOT`, `DATA_ROOT`.

`slurm/submit-training.sh` additionally takes `WATCH_ENV` (the existing evaluation
environment for the CPU monitors), optional `SETUP_JOB` (omit only when setup is
already verified), and optional space-separated `ARMS` (default `grpo tandem`,
use `grpo` to recover that arm alone). It submits a three-step
smoke and a 200-step full run for each arm immediately. GPU jobs depend on setup
completion when a setup job is supplied. Full jobs depend on monitor startup,
**not smoke success**. Each CPU
monitor checks the smoke's exit, every step's finite policy loss, the Tandem
senior fraction (0.3–0.7), held-out validation and saved safetensors shard lengths.
Failure or monitor termination cancels the associated full run. No automatic
replacement submission is performed. Full results remain provisional until all
smoke checks pass.

Smoke uses full response length and training batch settings, but only eight
held-out problems; it saves and validates at step 3. Full runs validate on all
1,000 held-out problems. Smoke and full paths are distinct. Per-attempt JSONL
metrics retain earlier observations after preemption; resumed steps supersede
the earlier observations. Files live under `RUN_ROOT/{grpo,tandem}-{smoke,full}`.

The two launchers now execute Python through uv and accept `TANDEM_ENV_FILE`
(default `train/env.sh`). The batch wrapper supplies `slurm/training-env.sh`,
with explicit site paths and no Conda dependency.

That environment file also disables Ray's automatic uv runtime-environment
rewriting, whose handling of `working_dir=None` fails with the pinned Ray.
Workers inherit the existing shared interpreter. `env/check_ray.py` exercises
real CPU Ray startup and verifies worker interpreter and patched module paths.

After both full runs and smoke checks finish:

```bash
uv run --python "$TANDEM_ENV/bin/python" --no-project python eval/select_figure2.py \
  --root "$RUN_ROOT" --out "$RUN_ROOT/models.json"
```

This rechecks the training artifacts and writes the official base plus the two
locally selected paths. Supply `MODELS_JSON=$RUN_ROOT/models.json` and the
separate evaluation environment to `slurm/figure2.sbatch`. Queue evaluation smoke
and full together with the existing CPU guard, using separate result directories.
See [FIGURE2.md](FIGURE2.md) for decoding, benchmarks and plotting.
