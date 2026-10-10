# Figure 2 reproduction audit — 2026-10-08 Pacific

## Reconnected scheduling audit — 2026-10-09 late Pacific

Engaging access is restored. GRPO word job 25374401 has 769/1,064 saved problems
and runs on non-preemptible `mit_normal_gpu`; Tandem 2608058 has 737/1,064 and
runs on CSAIL `vision-shared-l40s`. Final CPU graders still await completion.
Regrading GRPO's snapshot gives matched-737 macro word pass@8 of 58.44% versus
Tandem 58.72%, and pass@4 of 55.40% versus 55.39%. See
[matched word statistics](../results/figure2-regraded/word-matched-summary-20261009.json).

`sacct -D` distinguishes actual preemption from our four-hour continuation:

| Run | Actual PREEMPTED attempts | Planned REQUEUED attempts | Current restart count |
|---|---:|---:|---:|
| CSAIL Tandem 2608058 | 1 | 5 (about 3h55m each) | 6 |
| Engaging GRPO 25374401 | 2 | 4 (about 3h55m each) | 6 |

The planned requeues come from the five-minute pre-walltime signal handler.
They retain saved problems but repeat any uncommitted batch. Attempts to extend
the running allocations were rejected by both schedulers; their limits remain
four hours. Restarts must not all be described as preemptions.

Torralba was checked first at submission: H100 schedulable host RAM was then
insufficient, and all H200 GPUs were held by non-preemptible owner jobs. The live
recheck finds all eight H100 GPUs held by `vision-torralba-interactive` jobs and
all eight H200 GPUs by `vision-torralba-main`. Neither QoS is in main's preemption
list. Two shared jobs shown as dry-run preemption candidates request zero GPUs;
reclaiming them cannot free a GPU. The owner-route dry run gives a future start,
not immediate admission. The smaller Torralba hardware is incompatible with this
one-GPU BF16/model/cache requirement. The current shared allocation therefore
continues. Native requeue retains its shared route; it does not automatically
promote a job to owner QoS.

## Native-word status — 2026-10-09 late Pacific

Tandem 2608058 is still running, with six native restarts and 737/1,064 problems
saved. AIME, AMC and Minerva are complete; Olympiad is 254/581. Corrected final
grading 2608030 remains dependency-blocked. Paragraph grading 2608029 completed.
Engaging SSH authentication had expired and renewal subsequently encountered
connection refusal, so GRPO word jobs 25374401/25374417 could not be verified at
this check; their earlier status must not be presented as current.

The Tandem snapshot was regraded with the same corrected verifier. Word versus
paragraph handoff pass@8 is AMC 79.34% vs 78.51%, AIME 36.67% vs 38.89%, and
Minerva 61.40% vs 60.66%. On the same 737 problems, equal-benchmark macro pass@8
is 58.72% vs 58.98%. This is not a clear improvement, and does not answer whether
Tandem beats GRPO under word handoff. Word decoding uses training parameters;
the comparison is not a schedule-only ablation. Accepted word traces contain
10,086,766 tokens, 50.14% senior authorship, and 33.33% length-limited samples.

[Snapshot summary](../results/figure2-regraded/word-interim-summary-20261009.json)
records matched-subset metrics. The source snapshot and corrected sample scores
are retained separately under `results/figure2-regraded/`.

## Completion update — 2026-10-09 Pacific

All five original Figure 2 evaluations now contain all 1,064 problems and have
been regraded locally with the corrected shared grader. The final Tandem paragraph
file has eight attempts/problem and zero unfinished chains. Its scheduler job
later failed on a restart because the scratch checkout was unavailable on that
node; this blocked the dependent grading job despite the complete saved output.
After validating the artifact, dependency 2608029 was cleared so the cluster can
also retain corrected grades. No additional paragraph inference is needed.

[Complete corrected figure](../results/figure2-regraded/figure2-complete.png) ·
[Curve statistics](../results/figure2-regraded/figure2-complete.json).

| Macro metric | GRPO | Tandem |
|---|---:|---:|
| Solo pass@4 | 59.46% | 59.97% |
| Paragraph handoff pass@4 | 58.25% | 57.55% |
| Paragraph handoff pass@8 | 61.79% | 60.48% |

The extra native-word comparison is not finished. At this check Tandem 2608058
had 225/1,064 saved problems and GRPO 25374401 had 337/1,064; both were running,
with corrected CPU grading queued after completion. The older 992-problem
analyses below remain explicitly interim diagnostics.

The apparent Minerva deficit was mostly a grading artifact. After correcting
grading uniformly, the selected Tandem model is slightly stronger solo, but
paragraph handoff does not reproduce the paper's clear Tandem advantage. The
word-level training-schedule comparison is running; it has no result yet.

## Confirmed grading problem

Both reward files were identical to upstream commit
`334d02cf3fa2d0a5f6ac4a22709db040c7687426` before this correction. They were supplied
by the authors (with upstream Garena/Hendrycks/math-verify attribution), not
designed for this reproduction. Our setup omitted `math-verify==0.9.0` and
`latex2sympy2_extended==1.11.0`, despite both appearing in the authors' frozen
environment. The grader silently used a weaker fallback. Both training runtimes
and the checked evaluation runtime lacked these packages.

Concrete false negatives included `p(s)=s²+ω²` versus `s²+ω²`, `b=m` versus `m`,
and `Y(s)=1/(s+a)` versus `1/(s+a)`. Tandem more often boxed an equation with its
left-hand-side label, so the same defective grader could hurt the arms differently.

We separate two corrections:

1. **Restore the intended dependencies with unchanged author code.** Minerva solo
   pass@4 becomes GRPO 48.21%, Tandem 48.40%; solo pass@32 becomes 53.68%, 54.41%.
   Paragraph handoff pass@8 becomes 52.21%, 51.10%.
2. **Correct the remaining numeric-literal ambiguity.** The library interpreted
   a reference such as `2.55e-10` inside LaTeX as `2.55*e - 10`, not decimal
   scientific notation. Whole numeric literals on either side are now rewritten
   unambiguously, identically for every model and benchmark. The parse mode
   argument is also changed from a list to its required string. These are explicit
   modifications beyond restoring dependencies.

With both corrections, Minerva results are:

| Metric | GRPO | Tandem |
|---|---:|---:|
| Solo pass@1 | 49.31% | 49.21% |
| Solo pass@4 | 56.04% | 56.95% |
| Solo pass@32 | 61.40% | 63.60% |
| Paragraph handoff pass@4 | 57.77% | 57.96% |
| Paragraph handoff pass@8 | 60.29% | 60.66% |

These corrected scores must not be treated as directly comparable to published
absolute scores without confirming the authors' actual runtime. The frozen
environment is evidence of intended dependencies, not proof of what produced
each published curve. Regrading proves the false negatives and their effect on
our comparison; it does not prove the cause of the paper/reproduction difference.

## What remains different, and how much evidence supports it

| Explanation | Evidence and limits |
|---|---|
| Different training reward semantics | Confirmed dependency omission affected training, not just reporting. Correcting evaluation cannot reverse the resulting policy updates. The size of the training effect is not measured. |
| Checkpoint selection under the defective grader | Both validation and training used that grader. Selected checkpoints remain GRPO 160 / Tandem 180; they are not claimed to be optimal under corrected grading. Full validation traces were not saved (`validation_data_dir=null`). Tandem validation at steps 20 and 160 is also missing. |
| Training/evaluation schedule mismatch | Intentional in the authors' protocol: training uses stochastic word-boundary redraws; Figure 2 uses paragraph-level alternation. Matching paragraph code does not ensure that a newly trained policy transfers successfully. The requested native word test directly tests this distinction. |
| Limited samples and a single training run per arm | On 992 matched handoff problems, the macro Tandem-minus-GRPO pass@8 gap is −1.18 points, paired problem-bootstrap 95% interval [−3.34,+0.89]. It includes zero and excludes neither a modest gain nor loss. This interval is conditional on these trained checkpoints and does not measure training-seed uncertainty. |
| AIME coverage rather than per-draw accuracy | On all 90 AIME problems, GRPO has 189/720 correct attempts versus Tandem 187/720, nearly equal. But GRPO solves 39 distinct problems and Tandem 35: 33 shared, six GRPO-only, two Tandem-only. This produces the −4.44-point pass@8 gap; its paired 95% interval is [−11.11,+1.11]. |
| Response length / truncation | AIME paragraph traces average 2,635 GRPO versus 2,658 Tandem tokens; approximately 70.28% versus 71.39% reach 2,990 tokens by retokenization. Both are heavily budget-limited. Late-training tandem rollouts were longer and truncated more often (1,797 vs 1,563 tokens; 32.2% vs 24.6%). This is a plausible constraint, not proof that length causes the gap. The 3,000-token budget matches the protocol. |
| Data order, stochastic rollouts and interruption history | We explicitly use training/data seed 42; the released launchers leave the data seed unspecified. Native resume restores model/optimizer/RNG/dataloader state, but restarted rollout workers need not reproduce an uninterrupted stochastic stream. The author's selected Tandem step 120 differs from ours 180. These can change trajectories; none is established as the cause. |
| Domain specialization | Corrected base Minerva solo pass@32 is 67.28%, above both trained models, while training used competition math and Minerva includes undergraduate physics and differential equations. Forgetting/specialization is plausible for both arms; it no longer explains a large Tandem-specific deficit. |

The training audit found matching global batch 16, rollout group 8, PPO minibatch
8, one PPO epoch, learning rate 1e-6, clipping 0.2, no KL/entropy loss, no junior
gradient, word probability 0.5, 32-token redraw backstop, and matching dataset
hashes. Tandem's late-training senior-token fraction was 50.15%, with no observed
nonfinite metrics. This rules out several gross implementation failures, not
subtle optimizer/kernel/RNG differences. We have no evidence that GPU model alone
caused the result.

**Budget provenance:** the 3,000-token response cap in both training launchers
and `eval/config.py`, as well as the 4,096-token context limit and 8-token reserve
in `eval/common.py`, are present in upstream commit `334d02cf…`. We did not choose
these limits. The context-budget function is unchanged. Truncation is a possible
interaction between the trained policy and the shared budget, not an identified
budget mismatch with GitHub. Paper Table 3 also specifies 3,000 response tokens.
Tokenizing all prompts with the selected tokenizer confirms no budget reduction
for any AIME, AMC or Minerva problem. Only two Olympiad prompts have reduced
response allowances (the smallest is 2,792 tokens).

## Native-word execution check

The first full batch in 2608018 failed the per-token boundary check and was not
accepted as an evaluation result. Its 18,192-token cache could not accommodate
eight simultaneous full-length sequences. A source-level diagnostic of the
unchanged native sampler shows that `_word_select` drops state for requests absent
from the current batch, and `_word_replay` can choose a different author when a
request returns, even after a non-boundary token. This establishes a possible
mechanism, not proof that eviction caused that particular failed trace (the first
version did not save rejected samples).

Replacement 2608058 uses four concurrent sequences, which fit the measured cache,
and checks/saves the first complete problem before larger batches. It preserves
any rejected trace for diagnosis. Token budgets and the native sampler are
unchanged. Its first complete problem passed: eight attempts, 12,028 tokens,
49.37% senior authorship, with all switches validated. The GRPO counterpart was
then replaced by 25374401 using the same four-sequence execution settings.
The grading worktrees/environments remain separate. This finding concerns the
new word evaluation; it does not establish that this mechanism occurred during
training or explain the already-generated paragraph results.

## Corrected outputs and current jobs

[Corrected curves](../results/figure2-regraded/corrected-passk.png) use all solo
problems and 992 matched paragraph-handoff problems. AMC, AIME and Minerva are
complete; Olympiad is 509/581. Handoff has k=1,2,4,8; solo has
k=1,2,4,8,16,32, with no solo reference curves in the handoff row.
[Paired differences](../results/figure2-regraded/paired-differences.json) and
[AIME diagnostics](../results/figure2-regraded/aime-handoff-audit.json) are retained.
All source generations and original grades are preserved under
`results/figure2-regraded/source/`; corrected JSONs have source/dataset/grader
hashes and dependency versions. Minerva's environment-only diagnostic is
`results/figure2-progress/minerva-author-dependencies.json`.

The first required native-word runs use the same chosen checkpoints and frozen
base, the authors' patched sampler, Bernoulli 0.5 at word boundaries with a
32-token backstop, and training decoding 0.6 / 1 / −1. They use eight complete
attempts/problem. This jointly restores schedule and training decoding; it is
not a schedule-only ablation. A future `--decoding evaluation` run can separate
those factors. Every sample records token IDs and authorship for validation.

| Work | Cluster / job | State at launch |
|---|---|---|
| Selected Tandem word inference | CSAIL 2608058 | Running on one L40S; first problem validated; replaces failed 2608018 |
| Selected GRPO word inference | Engaging 25374401 | Pending; replaces cancelled 25372937 with corrected execution settings |
| Remaining Tandem paragraph inference | CSAIL 2607093 | Running; generation checkpoints retained |
| Correct final paragraph grades | CSAIL 2608029 | Depends on 2607093 and grading setup |
| Correct Tandem word grades | CSAIL 2608030 | Depends on 2608058; grading setup passed |
| Correct GRPO word grades | Engaging 25374417 | Depends on 25374401; grading setup 25373565 passed |

CSAIL grading setup 2608033 replaces failed 2608028, whose shared ANTLR cache
was damaged; the replacement bypassed that cache and passed. Engaging grading
successor 25373627 ended when its cancelled inference dependency was removed;
25374417 replaces it. Grading runs in separate CPU
environments and git worktrees so active generation jobs retain their original
code/environment. The corrected grader is applied after completion through
Slurm dependencies, without a polling timer. No additional training run has been
started, and no step-200 comparison has been resumed.
