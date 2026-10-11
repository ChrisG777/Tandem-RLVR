# Codex-judged elegance RL

The experiment adds a GPT-6.1-sol elegance bonus to full-weight Qwen3.5-9B RL,
using frozen Qwen3.5-2B for native word-level Tandem rollouts.

| Interface | Contract |
| --- | --- |
| [`prepare(archive, destination)`](../experiments/elegance/prepare.py) | Verified audit ZIP → disjoint train/validation/test Parquets and immutable split manifest |
| [`judge_solution(request, cache=...)`](../experiments/elegance/judge.py) | Question, candidate, answer and audited references → `{valid, elegant, reason}` through authenticated Codex CLI |
| [`compute_score(...)`](../experiments/elegance/reward.py) | Existing verl custom reward → correctness plus a 0.5 valid-elegant bonus, with separate diagnostic metrics |
| [`preflight`](../experiments/elegance/preflight.py) | Four live known-case checks → pass/fail receipt; failure blocks training |
| [`job.sbatch`](../experiments/elegance/job.sbatch) | Explicit cluster paths/resources → checkpoint-resuming Tandem or solo training |

The reward is `correct * (1 + 0.5 * valid_derivation * elegant)`. Correctness uses
the existing pinned math grader. Only correct answers invoke Codex. The judge
sees every valid audited reference and may accept a new valid shortcut. Neither
token count nor method-name matching directly affects reward. Bare answers,
omitted steps and instructions embedded in candidates cannot establish elegance.

Each CLI call is ephemeral, read-only, tool-disabled and schema-constrained.
The cache fingerprints the full request, rubric, schema, model and reasoning
effort. Successful grades commit atomically with CLI version and timestamp;
duplicate requests share a lock. At most four calls run concurrently across
reward workers. Infrastructure failures raise after bounded retries instead of
becoming incorrect answers. This uses Codex account limits, not a separate API key.

CSAIL calls use the existing `codex-cluster` launcher so SQLite stays on local
disk and its allocation-ownership guard remains enforced. It currently permits
one allocation to own the shared Codex runtime; independent simultaneous arms
need a separately authorized runtime or a shared grading service. The initial
launch is the Tandem arm. The CLI's ability to call this exact model is tested,
not inferred from native agent availability or authentication status.

## Fixed split

Hold out complement counting, Vieta and power of a point. Build a **joint** safe
training pool by intersecting their conservative training lists; exclude the
union of target/neighbor labels from every source route. Union their test IDs.
Seed 42 selects 20 in-family validation items from the safe pool, leaving 152
training and 17 test problems. Test never participates in training or checkpoint
selection. References are reward metadata, never part of policy prompts.

```sh
uv run --with-requirements env/grading-requirements.txt python -m experiments.elegance.prepare \
  --archive data/method-elegance/dataset.zip --out outputs/elegance/data
uv run --with-requirements env/grading-requirements.txt python -m unittest tests.test_elegance_reward
```

The wrapper reuses the validated FSDP2 CPU-offload and native Tandem implementation:
full senior weights, frozen junior, senior probability 0.5, word gap 32, batch 16,
minibatch 8, eight rollouts, 3,000 response tokens, learning rate 1e-6, seed 42,
and 200 updates. It permits enough epochs for the smaller dataset. Validation
runs every 20 steps. Recovery checkpoints are saved every five steps, with two
retained; only the final HF export is separately persisted to bound storage.
The initial campaign evaluates final weights; this is not validation-best
checkpoint selection. Run directories never reuse a different data manifest.

One 80-GB-class-or-larger GPU, 384 GiB host RAM (prior checkpoint peak), and eight
CPUs accommodate the existing trainer and concurrent grading subprocesses.
The job checks for 280 GiB of free checkpoint storage before loading models.
Preemption requeues and resumes optimizer/RNG/dataloader checkpoints, while
completed judge calls remain cached. Walltime continuation uses the existing
bounded runtime wrapper. Exact queue routes and live job IDs belong in the
campaign submission receipt rather than in launcher defaults.

## Initial launch: 2026-10-11

CSAIL job **2622847** started on `andreas-h100-1` using one H100, eight CPUs,
384 GiB host RAM, the 24-hour vision-shared route and automatic requeue.
Launch code is `bd84f67`. The live GPT-6.1-sol preflight passed all four cases
in 27.9 seconds: valid/elegant reference, valid/non-elegant reference, invalid
answer-only response, and invalid answer-only prompt injection. Four successful
judgments were durably cached before the training launcher started.

Campaign root:
`/data/vision/torralba/u/chrisge/tandem-rlvr/elegance-20261011`.
`submission.json` records exact submission arguments; `preparation-check.json`
records measured maximum prompt lengths of 369/143/205 tokens for
train/validation/test. `tandem/judge-preflight-2622847.json` contains the live
grader receipt. `tandem-2622847.out` is the job log. This launch record does not
assert that an optimizer update or final evaluation has completed.
