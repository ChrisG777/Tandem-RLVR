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
decoding. [LengthPenaltyRewardManager](../reward/length_penalty.py) subclasses
the manager to subtract the token cost, preserving `acc`, query accuracy and
format diagnostics. `PILOT_LENGTH_PENALTY=0.2` enables it in the existing
[training entry point](../slurm/shorthand-train.sbatch); omission preserves the
original reward. The adapter rejects invalid coefficients and response lengths.
Hydra records the coefficient, budget, manager and verifier paths per attempt.
Logs separately expose correctness reward, length cost, and response tokens.

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
