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

The site's Codex launcher was also corrected for same-job requeues onto a new
node (site tool commit `215af52`). The [preserved patch](../experiments/elegance/patches/codex-cluster-requeue.patch)
requires Slurm to confirm the current UID, running state, single allocated node,
batch host and positive restart count before migrating ownership. Other live
allocations remain excluded. All 12 existing storage tests and two focused
[requeue tests](../tests/test_codex_cluster_requeue.py) passed. This fixes the
grading runtime's restart path without bypassing its allocation guard.

## CPU-offload correction: 2026-10-11

The original job produced 23 rollout judgments but no completed optimizer update
in 36 minutes. The worker was computing; separate Qwen profiling identified
repeated synchronous gradient copies to CPU during accumulation as the bottleneck.
It was cancelled before submitting replacement **2623125**, which started on
`andreas-h100-1` with the same resources and campaign directory. Cached grades
and original logs are preserved. `restart-submission.json` records the replacement.

Launch code `070625c` explicitly loads an isolated verl checkout at `010d023a`,
based on the validated fork `48edde50`. The [patch](../experiments/elegance/patches/verl-deferred-sync.patch)
backports deferred gradient synchronization and honors the existing entropy
chunking option. Native fused CPU AdamW and a 10,000-token execution microbatch
budget reduce CPU overhead. The full-weight objective, batch/minibatch sizes,
rollout count, response limit, eager decoding, and data split remain unchanged.
The [parity check](../tests/test_fsdp2_offload_accumulation.py) previously produced
exactly equal gradients and AdamW updates for ordinary, deferred, and deferred
plus fused execution; separate full 9B eager training steps also passed. These
checks do not establish complete-run equivalence or final performance.

[`train/profile_qwen35.py`](../train/profile_qwen35.py) adds per-stage timings
under `tandem/profile/`; CUDA synchronization adds diagnostic overhead. Runtime
receipts now record resolved verl/vLLM module paths. The replacement passed all
four preflight cases from cache and loaded the isolated trainer. Completion of
its first update remains to be checked.

The replacement completed both optimizer calls (gradient norms 0.3308 and 0.2522)
with normal backward microbatches around 3–4 seconds and peak allocated GPU
memory 61.54 GiB. It then failed during vLLM weight wake-up with CUDA OOM, before
logging a complete RL step or saving a checkpoint. The trainer released unused
CUDA cache only after that wake-up. The [handoff patch](../experiments/elegance/patches/verl-rollout-cache-handoff.patch)
uses the existing `aggressive_empty_cache` helper before wake-up as well, preserving
live tensors and optimizer state. Patch application and Python compilation passed;
the live training-to-rollout transition still requires verification.

Job **2623300** replaces failed 2623125, using project `98d58f4` and isolated
verl `5807a7ec`. It started on `freeman-a100-1` with one 80-GB A100, the same
CPU/RAM request, and all four cached preflight cases passing. The additional
`VERL_LOGGING_LEVEL=INFO` setting records memory cleanup. Exact arguments are in
`handoff-restart-submission.json`. No complete RL step or recovery checkpoint
was produced by either earlier attempt; this restart replays the initial batch.

## Parallel senior-only RL: 2026-10-12

The user authorized the missing 9B solo arm in parallel with Tandem. Both use
full weights, the same base, split, reward, 3,000-token limit and 200 updates.
Solo clears native Tandem variables; only the senior generates and trains.
Raw rollout and validation records are saved for truncation diagnosis.

[`codex-runtime.py`](../experiments/elegance/codex-runtime.py) accepts only ephemeral
judge calls through the installed CSAIL allocation/storage launcher. Set
`ELEGANCE_CODEX_HOME` to an initialized private independent archive. This runtime
skips personal transcript imports, while retaining Slurm ownership exclusion,
node-local SQLite, snapshot recovery and same-job requeue checks. Credentials
and executable links stay outside git. Both arms share the original judge cache
and its four-slot concurrency pool. The running Tandem checkout is unchanged.
The job wrapper accepts `ELEGANCE_CODEX_BIN` to select this runtime.

The submission receipt is `solo-submission-20261012.json` under the campaign root.

Job **2629665** started on `beery-a100-1` (A100 80 GB), one GPU, eight CPUs,
384 GiB RAM, 24-hour vision-shared allocation with automatic requeue and saved
state resumption. Code `22b55a4`; all four reward tests and focused runtime
isolation checks passed. Cached preflight passed; startup was verified before
its first optimizer update. Tandem **2624597** continues independently.
At this audit Qwen3.5 solo/Tandem **2624812/2625164** and APPS coding
solo/Tandem **25639351/25639349** were all running. APPS is the coding campaign,
not a separate missing fourth pair. Both original reproduction arms completed.

## Judge failure recovery: 2026-10-12

Tandem job 2631898 hit a 30-second timeout in the launcher's `--version`
probe, before the actual judge call. That exception bypassed judge retries;
upstream verl accepted failed prompts and padded the incomplete batch. Step 66
contained 126 real responses and two empty padding rows. Updates 66–69 were
rejected. Their saved rollouts and job metrics were temporarily moved under
`tandem/rejected-after-step65-job2631898/`, then deleted at the user’s request
(including that directory). The committed checkpoint at restart remained 65.

[`judge.py`](../experiments/elegance/judge.py) now uses `run_codex(command,
prompt, timeout)` for both version probes and grading: it bounds process lifetime,
kills the process group on timeout, and normalizes launcher/exit failures into
retryable errors. `codex_version(executable, timeout)` caches successful probes
per worker. Failed judgments still never become cached or negative rewards.
The existing three-attempt retry budget and 300-second per-call timeout apply.

The [job wrapper](../experiments/elegance/job.sbatch) enables
`trainer.require_complete_training`: the [verl patch](../third_party/verl-tandem.patch)
rejects failed prompts, missing prompt groups, or missing/unexpected sessions
before padding and policy updates. Each new job runs four fresh preflight cases;
same-job requeues may reuse those verified grades. No reward or decoding change.

Nine focused failure/reward tests passed in the cluster environment. Replacement
job **2632052**, source **6f153bf**, uses isolated source at
`/data/scratch/chrisge/Tandem-RLVR-judge-fix`, full checkpoint 65, a 3,000-token
response limit, and one eligible 80-GB-or-larger GPU. Automatic requeue and resume
remain enabled. `judge-fix-relaunch-20261012.json` records the submission.
The other running arms retain their existing source versions.

## Usage and GPU-memory recovery (2026-10-12)

After the user reset Codex usage limits, both arms passed four fresh live judge
cases and resumed as Tandem **2635466** from 95 and GRPO **2635467** from 30.
Source **051e794** limits training and log-probability forwards to two examples.
Dynamic batching counted real tokens while the model allocated padded logits;
a mixed-length microbatch attempted an 86-GiB logits allocation. Fixed execution
microbatches prevent this expansion, while retaining the RL batch, optimizer
schedule, reward and 3,000-token response limit. Both arms now reject incomplete
training batches. Submission receipt: `recovery-20261012-usage-memory.json`.
