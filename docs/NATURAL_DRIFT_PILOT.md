# Natural language drift pilot

Correctness-only Solo RL on GSM8K, following the model/task choice in
[Sullivan & Koller (2026)](https://arxiv.org/abs/2610.02015). Their Llama/Gemma
results motivate the treatment models; Qwen is a useful lower-drift comparison.
This is a literature-informed pilot, **not an exact reproduction**: their linked
[code repository](https://github.com/coli-saar/language-drift) contained only a
TODO on October 10. The paper gives a search grid, not each winning configuration.

## Experiment

- Pinned **base** Llama-3.2-1B, Gemma-3-1B-PT, and Qwen2.5-1.5B. Revisions live
  in `data/build_drift.py`; no instruction-tuned initialization or synthetic code.
- GSM8K main: two fixed training examples used only as demonstrations;
  7,343 train / 128 validation / 1,319 official test. Seed 42; immutable hashes.
  Ordinary step-by-step prompt, ending in `####` numeric answer. No request for
  compression, jargon, secrecy, alternate language, or invented notation.
- 256 response tokens; prompt limit 1,024; group 64; temperature 0.5; 128 prompts
  per optimizer batch; one update per batch; two epochs / 114 full batches.
- Learning rate 1e-5, cosine decay, 3% warmup; clip lower 0.2 / upper 0.28;
  token-mean loss, group-standardized GRPO advantages, no KL/entropy/length reward.
  Final numeric correctness is the sole reward. First-answer accuracy is diagnostic.
- **Algorithm deviation:** existing verified verl trainer with DAPO-style asymmetric
  clipping and token loss, **without dynamic group resampling**. Do not label this
  full DAPO. Paper batch semantics and winning hyperparameters are unavailable.
  The two demonstrations differ from the paper's synthetic examples.
- Seed 42 first. Replicate a promising accuracy-preserving drift result in another
  seed before claiming reproducibility or launching matched Tandem comparisons.

## Interfaces and evidence

1. `uv run ... data/build_drift.py --out DATA [--model llama|gemma|qwen]` prepares
   hashed splits, optionally downloads pinned weights, and checks prompt lengths.
   Model access failures stop setup; never substitute a different model silently.
2. `slurm/drift-train.sbatch` takes explicit site paths, `MODEL_KEY`, `LR`, `SEED`.
   Uses existing Solo launcher, runtime, checkpoint persistence and bounded requeue.
   Model/tokenizer use plain completion with one BOS, not an instruct chat template.
3. Validation before training and every 10 steps; full rollouts and validation traces
   are saved by verl. Two full resume states retained; HF snapshots every 10 steps.
   Final step remains available in its actor checkpoint. Metrics are JSONL, not
   online W&B. The existing `eval/training_progress.py` reader can ingest logs;
   its Figure-2-specific plots should not be interpreted as GSM8K metrics.
4. `eval/drift.py --select-run RUN` selects highest greedy validation accuracy among
   saved checkpoints (earliest tie), writes `selected.json`, refuses silent fallback.
   `--model PATH --data DATA/test.parquet --out FILE` evaluates base or selected
   weights and saves resumable full traces plus first/last accuracy and truncation.

## What would count as success?

First confirm held-out final **and first-answer** correctness improve: simply
learning EOS is insufficient. Inspect matched base/early/best/final traces, including
correct responses at comparable lengths. Brevity, malformed output, arithmetic
mistakes and language mixing alone are not evidence of useful jargon.

Look for recurring expressions with consistent inferred meanings across held-out
problems. Use blinded human review; once candidates exist, test substitutions and
deletions against length-matched controls and cross-seed reuse. These semantic
interventions are a follow-up, not implemented metrics or evidence yet. A decrease
in reader legibility alone establishes at most a drift proxy. If accuracy collapses,
inspect earlier checkpoints instead of declaring gibberish successful jargon.

## Allocation and access

Initial estimate: one GPU with >=24GB, 48GiB host RAM, six CPUs; 2,048-token
training microbatches and rollout concurrency 32 bound memory. BF16/modern vLLM
excludes V100/TitanXP. Actual GPU startup remains to be verified. Use live policy
checks, not these estimates as a permanent default. Engaging checkpoints belong
on pool: scratch has only about 114GB free, pool about 357GB at preparation time.
Budget roughly 60GB per model including retained optimizer states and snapshots.

October 10 access check: Llama and Gemma return gated-model HTTP 403 on both
clusters; the local machine lacks access too. User was asked to enable access.
Qwen is accessible and serves as the comparison and end-to-end runtime check.
No duplicate cluster races, monitoring timer, or Re-ARC experiments.
