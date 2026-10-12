# MedCalc-Bench Tandem RLVR

This runs the paper's experiment on [MedCalc-Bench-Verified](https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified): train `Qwen/Qwen3-4B-Instruct-2507` with GRPO and with Tandem RLVR, then compare solo accuracy, reasoning-step handoff, and how predictable the senior's trace is to the frozen junior.

The training code is the repo's patched vLLM and verl stack. GRPO samples the senior alone. Tandem co-generates each rollout with a frozen copy of the base model, switching at word boundaries with probability 0.5, and the GRPO loss is applied only to senior tokens.

## What changes relative to the math run

- The reward is MedCalc's calculator check. Decimal answers must land in the row's lower/upper band (the dataset's 5% band). Integers, dates, and weeks/days follow the official checker. The model has to put the value in `\boxed{}`.
- The official test split (1,100 rows) is evaluation only. Checkpoint selection uses 1,000 rows held out of the official train split, stratified by calculator. The test set never selects a checkpoint.
- Prompts are longer than competition-math problems, so the prompt cap is 6,144 tokens. The response cap stays 3,000, matching the paper.
- Macro averages weight the seven calculator categories equally.

Hyperparameters that stay with the paper: learning rate `1e-6`, clip 0.2, batch 16, mini-batch 8, 8 rollouts, temperature 0.6, top-p 1, top-k -1, no entropy or KL penalty, 200 optimizer steps, validation every 20 steps, word handoffs, senior probability 0.5, 32-token gap cap, junior-token loss weight 0. Evaluation decoding is temperature 0.7, top-p 0.8, top-k 20. Solo reports pass@1 through pass@32 (32 samples). Handoff reports pass@1 through pass@8 (8 chains), switching on blank lines. Legibility is the junior's mean per-token cross-entropy on the senior's solo traces.

## Hardware

The paper trained Tandem on two A100 80 GB GPUs: the senior on GPU 0, the frozen junior on GPU 1. GRPO uses one GPU.

One A100 80 GB also works. GRPO runs as usual. Tandem colocates the junior on that same GPU and lowers vLLM's memory fraction so both KV caches fit. A 40 GB card is too small for single-GPU Tandem. Host RAM should be at least 128 GB because FSDP offloads the optimizer, and free disk should be at least 150 GB (model, dataset, and saved checkpoints).

`/dev/shm` should be at least 32 GB. Ray uses it. If you are in Docker, start the container with `--shm-size=32g`.

## Setup on the A100

The machine needs Linux, Python 3.11, an NVIDIA driver new enough for the CUDA 12.8 wheels (the math reproduction used a driver reporting CUDA 12.2), and network access for the first install.

```bash
bash experiments/medcalc/install_a100.sh
bash experiments/medcalc/prepare.sh
```

`install_a100.sh` creates `.venv-a100`, installs `experiments/medcalc/requirements.txt`, then overlays the patched vLLM and verl trees. `prepare.sh` downloads the pinned base model (`cdbee75f17c01a7cc42f958dc650907174af0554`), writes word-boundary token ids, and writes `data/medcalc/`.

Check the prompt-length line prepare prints. Training drops prompts longer than 6,144 tokens. If that share is large, raise `MAX_PROMPT` and `CONTEXT` together in `protocol.py` before training. `CONTEXT` must be at least the prompt cap plus 3,000.

## Train

Smoke is three optimizer steps and eight held-out problems. Run it before the 200-step jobs.

```bash
bash experiments/medcalc/train.sh grpo smoke
bash experiments/medcalc/train.sh tandem smoke
bash experiments/medcalc/train.sh grpo full
bash experiments/medcalc/train.sh tandem full
```

The same command resumes that arm. Full GRPO is one GPU. Full Tandem uses two GPUs when they are visible, and one 80 GB GPU otherwise.

After the tandem smoke, open the metrics JSONL under `checkpoints/medcalc/tandem-smoke/` and check `actor/tandem_senior_token_frac`. It should sit near 0.5. If the field is missing, the junior mask never reached the loss and the run is ordinary GRPO.

Selected weights are `checkpoints/medcalc/<arm>-<mode>/selected.json`. Selection is the best held-out pass@4 under that arm's own rollout (earliest step wins ties). The metric name is `val-core/medcalc/acc/best@4/mean`. If selection says the metric is missing, the JSONL key differed; rerun `train/figure2_checkpoint.py` with `--validation-metric` set to the key you see.

If a single 80 GB GPU runs out of memory on Tandem:

```bash
ROLLOUT_GPU_UTIL=0.22 TRAIN_BATCH=8 ROLLOUT_N=4 bash experiments/medcalc/train.sh tandem full
```

A shorter horizon is `MEDCALC_STEPS=50`. That is no longer the paper's 200-step run.

## Evaluate

```bash
bash experiments/medcalc/evaluate.sh check   # smoke checkpoints, 8 problems
bash experiments/medcalc/evaluate.sh         # full protocol
```

`check` only tells you the wiring works. The comparison to report is the full command: base, GRPO, and Tandem, each solo; GRPO and Tandem also hand off to the frozen base junior; then the junior scores each senior's traces.

Results land in `results/medcalc/full/`. The script prints a table and, when the runs are complete, a figure next to `summary.json`. Handoff retention is handoff pass@4 divided by that senior's solo pass@4. The paper's math result was about 94% retained for GRPO and about 100% for Tandem; MedCalc will not match those percentages, but the same comparison is the result.

`MEDCALC_SKIP_LEGIBILITY=1` skips the cross-entropy pass. Full solo evaluation is 32 samples on 1,100 notes, so leave several hours per model. Handoff on one GPU keeps both models resident; if that runs out of memory, rerun with `MEDCALC_GPU_UTIL=0.30`.

## Tests that do not need a GPU

```bash
python3 -m unittest tests.test_medcalc
```
