# Independent Figure 2 training

Both policies start from official `Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. The frozen Tandem junior is the same
base snapshot. No authors' RL-trained checkpoint is used for initialization,
checkpoint selection, or evaluation. Earlier released-model evaluation jobs
2559028, 2559029 and their monitor 2559040 were cancelled on 2026-10-05.

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

Each job reserves two H100/H200 GPUs and 192 GB host RAM. GRPO trains on both;
Tandem trains on one with its frozen junior on the other. Full jobs have a 24-hour
limit; the paper reports 7.8 h to GRPO step 200 and 9.4 h to Tandem step 120 on
two A100 80GBs. Actual runtime here remains unmeasured. At most three scheduler
restarts are allowed, resuming only that run's own optimizer checkpoint.

## Setup and interfaces

Clone the pinned upstream forks via SSH on CSAIL, then use
`bash third_party/apply_patches.sh` without `--install`.
`slurm/training-setup.sh` runs on a CPU allocation, creates a separate uv training
environment, installs the pinned dependencies and editable Python patches over
the matching vLLM binary wheel, runs tests, downloads only the base and builds
the common data split. Site inputs: `REPO`, `TANDEM_ENV`, `HF_HOME`,
`UV_CACHE_DIR`, `RUN_ROOT`, `DATA_ROOT`.

`slurm/submit-training.sh` additionally takes `SETUP_JOB` and `WATCH_ENV` (the
existing evaluation environment for the CPU monitors). It submits a three-step
smoke and a 200-step full run for each arm immediately. GPU jobs depend on setup
completion. Full jobs depend on monitor startup, **not smoke success**. Each CPU
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
