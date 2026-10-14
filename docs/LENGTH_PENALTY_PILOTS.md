# Length-pressure Solo RL pilots

Test whether a token cost induces reusable shorthand in RuleTaker shared
deductions and Reasoning Gym `manipulate_matrix`. Re-ARC is discontinued.

## Intervention and controls

Two new independent 100-update runs start from the same pinned
Qwen3-4B-Instruct-2507 base, not the already trained policies. Reuse the exact
datasets, seed 42, optimizer and rollout settings of the existing controls.

| Task | Response ceiling | Correctness reward | No-penalty control |
|---|---:|---|---|
| RuleTaker shared deductions | 4,096 | Fraction of four answers correct | `concept-pilot-20261008-budget-v2/ruletaker_shared`, 4K arm |
| Reasoning Gym matrix transformations | 3,072 | Exact final matrix match | `matrix-pilot-20261007`, 3K arm |

Training reward is **correctness − 0.2 × response_tokens / response_ceiling**.
Count all actual generated tokens, including the final answer and EOS, excluding
prompt and padding. Apply the cost to correct, incorrect and malformed outputs.
The maximum cost is .2, less than RuleTaker's .25 reward increment: a completion
with another correct answer always outranks one with fewer correct answers.
Among equally correct completions, shorter is better. GRPO still normalizes
advantages within each prompt group; the coefficient is not a guarantee of a
small policy change, particularly in groups with identical correctness.

The existing correctness verifier and verl reward manager handle grading and
decoding. [LengthPenaltyRewardManager](../experiments/shorthand/length_penalty.py) subclasses
the manager to subtract the token cost, preserving `acc`, query accuracy and
format diagnostics. `PILOT_LENGTH_PENALTY=0.2` enables it in the existing
[training entry point](../experiments/shorthand/slurm/shorthand-train.sbatch); omission preserves the
original reward. The adapter rejects invalid coefficients and response lengths.
Hydra records the coefficient, budget, manager and verifier paths per attempt.
Validation logs separately expose correctness reward, length cost, and response
tokens. Training logs currently expose only their combined reward and mean
response length; per-prompt training rollouts are not retained.

Keep prompts unchanged: step-by-step reasoning plus a final answer block, no
supplied jargon. Batch 16 prompts, eight rollouts, minibatch eight, temperature
.6, LR 1e-6, no KL/entropy bonus. Save every ten updates, validate every 25,
retain final step 100. Validation reports correctness separately from the
penalized reward; held-out evaluation uses the original correctness-only grader.

## Evaluation and interpretation

Queue checkpoint verification followed by standard and longer-composition test
evaluations, four completions per problem, seed 17. Evaluate RuleTaker at 4K/8K
and matrix at 2K/3K, matching existing baseline files. Reuse base evaluations.
The existing review tool creates uniform and both-correct blinded pairs.
Compare the new runs against both base and the matched no-penalty trained arm.

Report exact accuracy, RuleTaker query accuracy, tokens, truncation and format
alongside transcript inspection. Shorter English, omitted reasoning, ordinary
rule numbers and standard matrix notation do not alone establish emergent jargon.
Look for recurring conventions with stable meaning across held-out tasks.
This first pilot has one seed per condition and one penalty strength; it does
not establish a causal mechanism or absence of jargon from a few examples.

## Allocation and recovery

Reuse calibrated datasets and existing runtimes. One >=80 GB compatible GPU,
six CPUs and 144 GiB host RAM per training run; checkpoint verification is CPU
only; evaluation uses one >=24 GB compatible GPU, four CPUs and 32 GiB RAM.
Training and evaluation preserve existing bounded same-job requeue/resumption.
Separate campaign directories prevent mixing objectives or overwriting controls.
No cluster races or monitoring timers. Record live site selection and job IDs
below after submission.

## Submitted 2026-10-09, 23:29 Pacific

Frozen code: `a1acb36`, detached worktree `Tandem-RLVR-length-20261009` on
each cluster. No existing experiment was cancelled or modified.

| Task / cluster | Train | CPU verification | Evaluation | Review vs base | Review vs no-penalty RL |
|---|---|---|---|---|---|
| RuleTaker / CSAIL | `2613501_0` | `2613502_0` | `2613503_0` | `2613504` | `2613505` |
| Matrix / Engaging | `25474790_1` | `25474791_1` | `25474792_1` | `25474793` | `25474794` |

At 23:30 Pacific, matrix training was running on `node2103` (A100 80 GB);
RuleTaker was pending for priority. Downstream jobs depend on successful
training/checks. RuleTaker's no-penalty comparison also waits for the already
running control evaluation `2613450_0`. Each review contains four comparisons
(two splits × two evaluation budgets), including uniform and both-correct pairs.

Campaign directories:

- CSAIL: `/data/vision/torralba/u/chrisge/tandem-rlvr/length-pilot-20261009/ruletaker_shared`
- Engaging: `/orcd/scratch/orcd/013/cge7/tandem-rlvr/length-pilot-20261009/manipulate_matrix`

Each stores copied base/calibration provenance, a fresh dataset gate receipt,
`jobs.txt`, checkpoints, evaluations, and `comparison-to-control/` for the
direct no-penalty comparison. Base/control evaluation symlinks reuse artifacts;
the original datasets and scores were not changed.

Allocation: CSAIL owner dry run forecast a material wait behind existing owner
work; neither shared route forecast immediate admission. RuleTaker therefore
uses all compatible vision-shared H100/H200/A100-80 partitions, per policy.
Dependent evaluation admits all compatible 24+ GB Ampere-or-newer vision-shared
types. Engaging combines normal/preemptible routes and excludes incompatible
nodes using the saved live inventory and [ORCD VRAM mapping](https://orcd-docs.mit.edu/running-jobs/available-resources/).
A hostname-feature OR list exceeded a scheduler limit during dry run, so the
accepted request uses compressed exclusions instead. Both sites use Nice 0.

Engaging reported 786.7/1,024 GB scratch usage (independently measured 787 GiB)
before submission. The existing comparable run occupies 123 GiB; one new run
fits with additional room for checkpoint replacement writes. CSAIL had 608 GiB
free. These are preflight measurements, not reservations against other work.

Validation: four actual-verl reward integration tests passed independently in
both installed training environments. They cover token/padding accounting,
correctness ordering, matrix/malformed answers, unchanged accuracy diagnostics,
zero/invalid coefficients, and assembly into token rewards. Four original
reward/dataset tests passed locally; both full calibration/data hash gates
passed; Hydra accepted the new configuration and shell syntax checks passed.

## Collapse investigation (2026-10-10)

Both 100-step runs and their evaluation chains completed. At matched evaluation
budgets, matrix test accuracy fell from 92.38% (no-penalty 3K-trained control)
to 15.63%, with mean length 1,889 to 678 tokens. RuleTaker at 4K fell from
79.49% to 62.11% all-four accuracy (92.29% to 86.91% per-query), with mean
length 2,377 to 1,290. These are fixed final-step comparisons, not selected-best
checkpoints. RuleTaker remains above the base model's 43.55% all-four accuracy.

The leading explanation is amplification of length pressure by per-prompt GRPO
standard-deviation normalization. The saved configurations use
`algorithm.norm_adv_by_std_in_grpo=True`, token-mean loss, no reference KL,
and zero entropy bonus. The installed
`verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` computes
`A = (R - group_mean(R)) / (group_std(R) + 1e-6)` and broadcasts it across
response tokens. When correctness is constant within a group, substituting
`R = correctness - lambda * length / budget` cancels lambda and budget,
approximately giving `A = -(length - mean(length)) / std(length)`.
Thus a small cost can produce an order-one shortening signal, including positive
advantages for short incorrect answers in all-wrong groups. Correctness ordering
within a group is preserved; its priority across different prompt groups is not.

A CPU check using the actual installed GRPO function on Engaging, with eight
lengths 1,000 through 1,700 in increments of 100 and budget 3,072, confirmed:

| Equal correctness in group | Coefficient | Shortest advantage | Longest advantage |
|---|---:|---:|---:|
| All incorrect or all correct | 0 | 0 | 0 |
| All incorrect or all correct | .2 | +1.4288 | -1.4288 |
| All incorrect or all correct | .02 | +1.4280 | -1.4280 |

This is a synthetic mechanism check, not a reconstruction of training groups.
Simply lowering lambda tenfold therefore does not reliably weaken the signal
in equal-correctness groups.

Observed evidence:

- Matrix validation accuracy at steps 25/50/75/100 was
  84.38/77.34/51.95/16.99%; validation penalized reward also fell from .7505
  at step 25 to .1273 at step 100. The collapse did not improve the intended
  scalar objective, and is not merely a test-time evaluation failure.
- In the 1,024 matrix test completions at 3K, 476 penalty-run final answers
  had a row/column shape different from their target, versus 24 control answers.
  Inspected traces drop rows during transformations. These are actual content
  errors rather than only shorter English. No truncations occurred in this
  penalty evaluation.
- The final matrix evaluation had all four completions wrong on 202/256
  problems, versus 5/256 for the control. Evaluation groups have four samples;
  they are not the eight-sample training groups and cannot measure historical
  training advantages.
- Mean actor entropy over steps 81–100 was .01045 with the penalty versus
  .03043 in the control; mean gradient norm was 1.127 versus .233. These
  support a stronger, less diverse update regime but do not establish causality.
- RuleTaker validation query accuracy peaked among recorded checks at 90.14%
  at step 50 and ended at 87.01%, while length continued falling. Its dense
  partial-credit reward is a plausible reason for milder degradation than the
  matrix exact-match reward, not a demonstrated explanation.

Checks against alternative explanations: all 100 matrix steps came from one
uninterrupted post-preemption attempt (the first attempt logged no updates).
All recorded numeric metrics in both tasks' treatment/control runs were finite.
Matrix control/treatment used the same base revision, learning rate, batch sizes,
temperature, and microbatch token budget; their effective launch overrides differ
only in the reward adapter/coefficient and artifact/data paths. Matrix train,
validation, and test files and the correctness verifier were byte-identical
between the inspected worktrees. All four actual-verl reward integration tests
were rerun successfully, checking padding, token counting, correctness ordering,
and reward assembly. No reward-accounting bug was found in these checks.

Limits and next experiment: per-group training rewards/lengths were not saved,
so the historical fraction of length-only updates cannot be recovered. Early
model weights were pruned (the inspected run retains steps 90 and 100), although
their validation metrics remain. A causal test should repeat both zero-penalty
and penalty conditions with standard-deviation normalization disabled, keeping
other settings fixed, and log per-group correctness/length/advantages. Preserve
validation checkpoints and stop on sustained accuracy degradation. This proposed
ablation has not been submitted; no existing jobs were changed in this audit.
