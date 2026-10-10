# Reusable-concept Solo RL pilots

Two independent task pilots test whether repeated reasoning induces reusable
conventions beyond shorter English. These are adapted synthetic tasks, not
claims of results on the original RuleTaker or ARC benchmarks.

## Data and rewards

| Task | Repeated structure | Answer/reward | Held-out extension (`long`) |
|---|---|---|---|
| `ruletaker_shared` | Five entities; eight branching Horn rules with shared intermediate conclusions; four related entailment questions per completion | Four true/false values; reward = fraction correct; `acc` = all four correct | Seven entities, same rule motifs |
| `rearc_objects` | Three input/output demonstrations and one query from a hidden object-transformation family | Exact output grid, binary reward | Query grid has largest side 8–9 instead of 5–7; demonstrations unchanged |

RuleTaker-derived data use the upstream `Fact`, `Rule`, and `Theory` records and
English renderers from [allenai/ruletaker](https://github.com/allenai/ruletaker)
commit `abaacec9364992eff5ec4555b837e20fee2f2ff0`. Our adapter constructs a
branching rule graph; ProbLog 2.2.10 labels entailment, with explicit closed-world
instructions. Entities, property names, relations, facts and sentence/query
order vary. Each split has 50% true query labels by construction; sampling
rejects theories that cannot provide the prescribed labels. The graph topology
is shared across splits: this pilot tests reuse of those motifs, not unseen
logical forms. A completion receives reward immediately after its four answers.

Re-ARC uses paired generators/verifiers from pinned
[Reasoning Gym](https://github.com/open-thought/reasoning-gym/tree/49b07130b3fcd12f2d064bba7c43869543a0e7e7),
which incorporates [Re-ARC](https://github.com/michaelhodel/re-arc).
Four equally represented families: enclosed-region filling (`00d62c1b`), filling
an object's bounding rectangle (`6d75e8bb`), moving an object inside corner
markers (`a1570a43`), and directional coloring around objects (`d364b489`).
Generator difficulty interval is [0, 0.3]; reject unchanged grids, verifier
mismatches, and grids outside the size limits. Minimum side is four cells.
Family IDs and verifier programs are metadata only, never prompt content.
These public task families may be familiar from pretraining; the pilot does not
establish that an observed convention is novel without inspecting base traces.

Each task has calibration 64, train 4,096, validation 128, test 256, and `long`
256 instances. Seeds are 42 plus a distinct million per split, then candidate
index. All prompts fit 1,024 tokens with the pinned Qwen tokenizer. Deduplicate
prompts globally and also entire RuleTaker theories / all Re-ARC demonstration
and query inputs across splits. Every ARC output is checked against its paired
verifier. Manifests include source versions and Parquet hashes. Only calibration
scores may inform difficulty revisions; test/long labels are used for integrity
checks, not model-based selection.

## Training and gates

Revised budgets, **4,096 and 8,192 response tokens**, start separately from
Qwen3-4B-Instruct-2507. Each Solo GRPO run uses 100 optimizer steps, seed 42,
16 prompts/update, eight rollouts/prompt, minibatch eight, temperature 0.6,
learning rate 1e-6, no KL or entropy bonus, microbatch token budgets 5,120 / 9,216 respectively (prompt plus response).
The effective prompt/minibatch sizes stay fixed; the larger microbatch ceiling
allows a single long trajectory to fit. Evaluation context is 10,240 tokens.
Checkpoint every ten steps; validation every 25; retain the fixed final step-100
HF model. There is no length penalty, shorthand reward, supplied vocabulary,
or supervised jargon example. Evaluation samples four completions per prompt
at each budget (seed 17), for both base and both trained checkpoints.

Training runs only after calibration passes: generous-budget exact accuracy in
(0, .98), at least 90% final answer blocks, **less than 10% truncated**, and at
least one shorter-budget exact success. RuleTaker query accuracy must exceed
55% (random baseline 50%); every Re-ARC family must have a generous-budget
success. A failed gate stops that task's downstream chain, independently of the
other task. This guards against another truncation-driven pilot; it does not
prove that 8,192 tokens are unconstrained for every example. Revisions must be
saved as new campaigns, not overwrite prior evaluation provenance.

Compare exact accuracy, RuleTaker query accuracy, token lengths and truncation,
then inspect blinded paired traces including the both-correct subset. Count a
candidate convention only when a stable meaning recurs on held-out problems;
ordinary variable names, ARC coordinates, prompt vocabulary, and shorter English
alone are not evidence of emergent opaque jargon. Frozen-reader original versus
English-expansion tests are a follow-up, not implemented by this pilot chain.

## Interfaces and execution

- `build_dataset(task, out_dir, seed, ruletaker_source)` in
  [the builder](../data/build_concept_pilot.py) writes immutable splits using
  upstream oracles; it rejects revision drift and existing output directories.
- [Reward](../reward/shorthand_reward.py) retains binary `acc`; fractional
  RuleTaker `score`/`query_accuracy` never changes exact `pass_at_n` semantics.
- [Submit](../slurm/submit-concept-pilot.sh) takes one task, existing calibration
  job ID, prepared paths and live-validated CSAIL partition lists; it submits
  two training jobs → CPU checks → matched evaluations → blinded review, plus
  base evaluations. Its submission lock prevents duplicate chains.
- [Calibration check](../train/check_shorthand.py) verifies scores, all data
  hashes and prompt limits before writing `calibration/verified.json`.

Generate with `uv run --project env/shorthand-data python data/build_concept_pilot.py
--task TASK --out-dir data/concept-pilot/TASK`; RuleTaker additionally needs
`--ruletaker-source PATH` to the pinned upstream checkout. The dedicated uv
project is used only for data preparation, not cluster training environments.

GPU jobs use one GPU; inference admits >=24 GB Ampere-or-newer GPUs,
32 GiB host RAM/four CPUs;
training >=80 GB, 144 GiB/six CPUs, based on the completed matrix pilot.
Native requeue resumes saved checkpoints or atomic evaluation batches, with at
most 12 restarts and a five-minute pre-walltime signal. Jobs use six-hour chunks,
except two-hour calibration. CPU integrity/review jobs request no GPUs.
New CSAIL pilot jobs use Nice 0 following withdrawal of the temporary priority penalty. No
cross-cluster duplicates or monitoring timer are created.

Inference's 24 GB eligibility is a memory estimate: roughly 7.5 GiB BF16 weights,
0.56 GiB KV per full 4,096-token sequence, plus activation/runtime space;
vLLM sizes and schedules its KV cache within 75% of VRAM, with eager execution
and 4,096 batched tokens. This admits available Torralba RTX 3090s for calibration.
V100/Turing cards are excluded from this BF16 runtime. Training keeps its measured
80 GB minimum. Dependent evaluations admit all compatible vision-shared types.

The initial and revised submission records are below.

Storage planning: each completed matrix arm currently occupies 123 GiB including
retained restart checkpoints and final weights. Four comparable arms would use
about 492 GiB, versus 860 GiB free on the CSAIL Torralba filesystem at preflight.
This is an estimate, not a quota reservation; all four use that filesystem rather
than Engaging's more constrained scratch/pool allocations.

## Submitted 2026-10-08, 19:49 Pacific

Code/data are frozen at `96fd6d4` in the detached CSAIL worktree
`/data/scratch/chrisge/Tandem-RLVR-concept-pilot-20261008` (no additional branch).
Artifacts live under
`/data/vision/torralba/u/chrisge/tandem-rlvr/concept-pilot-20261008/{TASK}`.

| Task | Calibration | Solo array (1,024 / 3,072) | CPU verification | Base evaluation | Trained evaluation | Review |
|---|---|---|---|---|---|---|
| RuleTaker | 2607129 | 2607130_0 / 2607130_1 | 2607131_0–1 | 2607132 | 2607133_0–1 | 2607134 |
| Object Re-ARC | 2607135 | 2607136_0 / 2607136_1 | 2607137_0–1 | 2607138 | 2607142_0–1 | 2607143 |

Calibration uses the compatible Torralba owner route (RTX 3090/H100/H200);
training uses all compatible vision-shared H100/H200/A100-80 nodes. Dependent
inference uses all compatible vision-shared 24+ GB Ampere/Ada/Hopper types.
Submission checks confirmed owner H100/H200 GPUs were occupied by
non-preemptible owner jobs, while Torralba RTX 3090 capacity was available.
Every downstream job is gated by successful upstream completion; array elements
use corresponding-element dependencies. Training/evaluation restart support is
in the existing runtime, with pilot-specific evaluation coverage added here.

Validation: 13 focused checks passed before submission, including rewards,
calibration gates, all split hashes/oracle labels, and host-side restart behavior.
No training result or jargon claim is available at submission time.


### Budget revision after calibration

The initial 1,024-token RuleTaker calibration had 100% truncation and no final
answers. Re-ARC at 3,072 tokens had 96.1% truncation, 3.9% final answer blocks,
and 0.78% exact accuracy. These are calibration observations, not test results.
The initial training, verification, base-evaluation, trained-evaluation and review
chains in the table above were cancelled while still pending; no RL was run at
those budgets. Both original calibration jobs were allowed to finish. Re-ARC
calibration 2607135 moved in place to vision-shared after the Torralba owner
capacity filled and subsequently ran on andreas-h100-1.

The revision keeps the same prompts/data/rewards, expands response limits to
4,096 / 8,192 and retains the <10% generous-budget truncation gate. The existing
evaluation interface now accepts an explicit context length and records it in
provenance; defaults for reproduction and previous pilots stay at 4,096. New
training uses maximum prompt length 1,024. The long arm's memory use is an
estimate until it starts; larger microbatches may require revising the GPU
request if the first updates expose a memory limit.

### Revised submission, 20:00 Pacific

Frozen code `c145e48` is in
`/data/scratch/chrisge/Tandem-RLVR-concept-pilot-20261008-v2`.
Artifacts use
`/data/vision/torralba/u/chrisge/tandem-rlvr/concept-pilot-20261008-budget-v2/{TASK}`.
Data hashes are unchanged. All GPU jobs use the compatible vision-shared route
at submission because the compatible Torralba nodes were occupied and neither
shared-route dry run established immediate admission. Nice remains 1000.

| Task | Calibration | Solo array (4,096 / 8,192) | CPU verification | Base evaluation | Trained evaluation | Review |
|---|---|---|---|---|---|---|
| RuleTaker | 2607187 | 2607188_0 / 2607188_1 | 2607189_0–1 | 2607190 | 2607191_0–1 | 2607192 |
| Object Re-ARC | 2607193 | 2607194_0 / 2607194_1 | 2607195_0–1 | 2607196 | 2607197_0–1 | 2607198 |

The original RuleTaker calibration confirmed RTX 3090 compatibility: measured
7.61 GiB model memory and 9.4 GiB available KV cache (68,448 tokens). A larger
per-request context reduces maximum concurrent sequences within that same cache.
Fifteen focused tests pass after the context change, including interruption/resume
at an 8,192-token response budget and rejection of mismatched context provenance.
The four revised training arms remain conditional on calibration success.

## Status and evaluation recovery, 2026-10-09

Both RuleTaker arms reached step 100 and passed direct checkpoint/metric
verification. Final validation (128 problems, four samples/problem):

| Training response budget | All four questions correct, mean over samples | Individual-question accuracy |
|---|---|---|
| 4,096 | 83.79% | 93.90% |
| 8,192 | 79.88% | 91.75% |

Training mean reward rose from 74.16% to 91.54% for the 4,096-token arm and from
81.64% to 91.76% for the 8,192-token arm (first versus last ten updates).
Mean response length changed from 2,697 to 2,419 and from 2,825 to 3,052 tokens,
respectively. These are training/validation summaries, not matched test-set
improvements or evidence of opaque jargon.

The CPU verification jobs 2607189_0/1 failed before Python started: their uv cache
fell back to the AFS home, which was inaccessible after credential expiry.
Consequently the trained evaluations and review were cancelled by dependencies.
The launcher now sets shared cache paths explicitly. Both verification checks
were rerun successfully against all 100 observed steps and the final HF weights.
A separate compatibility correction makes the calibration prompt-length check
request token IDs explicitly (`return_dict=False`); Transformers 5 returned a
mapping whose length had incorrectly been recorded as two. Stored dataset and
evaluation token lengths show the actual RuleTaker prompts fit the configured
limit; this correction does not change training data or weights.

Replacement trained-evaluation array **2613450_0/1** uses the original frozen
`c145e48` evaluation code and identical budgets, context, data and models.
Review **2613451** depends on both evaluations. Completed base evaluations are
reused. The existing 32 GiB/four-CPU, single-GPU inference shape passed an owner
route dry run; compatible Torralba RTX 3090/H100/H200 partitions are admitted.
These jobs use Nice 0 in accordance with the updated shared allocation policy.
No RL training was restarted and no Re-ARC gate was bypassed.

Re-ARC did not enter RL: at 8,192 tokens calibration accuracy was 16.80%, only
36.72% emitted a final answer, and 63.28% truncated. Family accuracies were
4.69% (enclosed regions), 3.13% (bounding rectangle), 0% (corner-marker placement),
and 59.38% (directional coloring). The gate blocked this mixture because it
would retain the severe truncation confound. No jargon conclusion is available
for either new task pending matched evaluation and semantic review.
