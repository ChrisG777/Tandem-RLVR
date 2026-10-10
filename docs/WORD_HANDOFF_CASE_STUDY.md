# Word-handoff error attribution: initial case study

Manual review on 2026-10-10 finds errors introduced by both the frozen base and
the trained senior, plus failures caused by unfinished or mispackaged answers.
This small, deliberately selected sample does not establish which model usually
introduces errors, or a recovery advantage for tandem training.

## Data and selection

Used the existing local regraded snapshot: GRPO step 160, Tandem step 180,
32 solo samples per problem, and 8 native-word samples per problem. Word results
cover 769 GRPO problems and 737 Tandem problems, with 737 in common. These are
partial October 9 snapshots, not newly fetched completed evaluations.

Ranked problems by the larger of the two within-senior solo-minus-word success
rate drops, then selected interpretable cases for manual inspection. No problem
had both solo models at least 75% correct and both word conditions at most 37.5%
correct in this snapshot. The cases below therefore include asymmetric drops.
The arithmetic case received a complete review of all 16 handoff traces;
the other cases are supporting diagnostics, not a complete attribution census.

| Problem ID / task | GRPO solo | GRPO + base | Tandem solo | Tandem + base |
|---|---:|---:|---:|---:|
| `amc23_25-103`: solve 57x + 22y = 400 | 27/32 | 5/8 | 23/32 | 2/8 |
| `amc23_25-39`: area of four diamonds | 32/32 | 4/8 | 21/32 | 4/8 |
| `minerva-64`: population ODE | 32/32 | 1/8 | 2/32 | 1/8 |
| `amc23_25-4`: table-tennis tournament | 26/32 | 6/8 | 32/32 | 2/8 |

These are saved grader outcomes, not new semantic grades. Solo decoding is
temperature 0.7 / top-p 0.8 / top-k 20; word handoff uses 0.6 / 1 / -1.
Consequently these drops are not a controlled schedule-only effect. Sample
indices are zero-based and are not paired continuations across conditions.

## Arithmetic: direct evidence against assuming the base is always responsible

For positive integers satisfying 57x + 22y = 400, x is between 1 and 7 and
13x = 4 modulo 22. Thus x = 2, y = 13, and the answer is 15.
An independent integer enumeration verifies the unique solution.

The traces usually use the longer extended-Euclidean route:
1 = -5(57) + 13(22), then scale to (-2000, 5200). A frequent mistake is
dropping zeros during this multiplication and propagating the wrong particular
solution. Another trace correctly scales and then reuses the unscaled pair.

| Senior / sample | First substantive error | Author of decisive erroneous content | Subsequent outcome |
|---|---|---|---|
| GRPO 1 | Scales -5 to -200 instead of -2000 | Base | No repair; stops with boxed 1815 |
| GRPO 2 | Scales -5 to -200 | Senior | Finds (2,13) by enumeration, then returns to bad scaling; truncates |
| GRPO 6 | Scales -5 to -20 | Senior | Corrects scaling and computes (2,13), but truncates during verification |
| Tandem 1 | Writes 57 = 2 × 22 + 11 | Senior | Repairs remainder; base then introduces bad scaling; scaling is eventually repaired, but trace truncates |
| Tandem 3 | Scales -5 to -20 | Base | Detects contradiction but never stabilizes correction; boxed 185 remains |
| Tandem 4 | Scales -5 to -200 | Base | Finds (2,13), returns to bad scaling, truncates |
| Tandem 5 | Reuses (-5,13) after already computing (-2000,5200) | Senior | Continues through contradictions and new scaling errors; truncates |
| Tandem 6 | Scales -5 to -200 | Senior | Finds (2,13), retains erroneous scaling, truncates |
| Tandem 7 | Scales -5 to -200 | Base | Finds (2,13) through congruences, returns to bad scaling, truncates |

Among these **nine failed traces, four first substantive errors are base-authored
and five senior-authored**. GRPO contributes 1 base / 2 senior; Tandem contributes
3 base / 3 senior. This is a description of one selected problem, not an estimate
over the benchmark. Closely related attempts are not independent evidence about
the population of mathematical errors.

“Substantive” excludes incidental prose such as saying both coordinates are
negative when one is positive, if subsequent constraints use the correct values.
Attribution is to the decisive wrong number or mathematical choice, not all words
in its sentence. Exact surrounding spans are preserved in the evidence file.
This locates the emitted mistake; it does not prove counterfactual responsibility
for a jointly conditioned trajectory.

## Recovery is often joint, and can happen without a passing answer

GRPO sample 3 is a useful successful control. The **senior** writes
`57(-200) + 22(520) = 400`. Later the **base** writes the corrected `-2000`,
and the **senior** writes the corrected `y_0 = 5200`. The trace finishes with
boxed 15. Thus the reverse direction—base participation in repairing a senior
error—is visible too.

Tandem sample 1 provides a partial rescue after a base error. The base emits
`57(-20)` while multiplying by 400. Much later, the base writes `57(-2000)`
and the senior writes `22(5200)`. The trajectory finds t = 91 but hits the
3,000-token limit before completing its answer. Calling this either “no recovery”
or “the senior saved the trace” would lose important information.

GRPO sample 6 also fixes its scaling and reaches (2,13), but truncates before
boxing 15. Other traces rediscover the valid solution while retaining a
contradictory derivation. Useful labels are therefore: error detected, erroneous
state corrected, valid solution found, correction retained, and final answer
accepted. These should be measured separately.

There is **no demonstrated tandem-trained rescue advantage in this pilot**.
GRPO has one passing recovery from a substantive arithmetic error; Tandem has
none on this problem. The inputs and errors differ, so this is not a fair causal
comparison of recovery ability.

## Other large drops include different failure mechanisms

**Area, `amc23_25-39`.** The first-quadrant region is a diamond centered at
(1,1), with diagonals 2 and 2 and area 2. Reflection gives total area 8.
Both seniors pass 4/8 handoff attempts. All eight failed attempts hit the token
limit and contain no boxed answer. Inspected traces repeatedly reconsider
whether the quadrant area should be multiplied by four, sometimes after already
obtaining 8. For example, Tandem sample 1 repeatedly reaches 8 but never boxes
it; sample 3 spends substantial text disputing correct triangle vertices.
This is a loss of convergence/termination as well as a mathematical-error issue.
It is not enough to assign the whole failure to whoever wrote the last token.

**ODE, `minerva-64`.** The requested equation is dx/dt = kx - a.
The inspected failed examples write this correct equation, then solve it and box
x(t) = a/k + C exp(kt), or its initial-condition form, instead. All seven failed
attempts per senior stop normally and their boxed answers are solution functions;
the one passing attempt per senior boxes the ODE. This is an answer-target /
extraction mismatch, not evidence that the core differential equation was
misderived. Some examples also append incorrect stability commentary, so they
should not simply be relabeled wholly correct. The requested equation and the
boxed deliverable need separate semantic labels.

**Tournament, `amc23_25-4`.** All eight failed handoff attempts across the two
seniors hit the token limit. In GRPO sample 0, the trace first boxes the correct
36, later boxes an infeasible alternative 276, and introduces a mistaken
cross-group win accounting equation. GRPO sample 5 eventually rules out the
alternative but never boxes its answer before truncation. This illustrates why
“ever found the answer,” “last extracted answer,” and “valid completed argument”
are distinct outcomes. No model-level attribution census was done for this task.

## Snapshot-wide token-limit diagnostic

| Senior | Failed attempts on 737 matched problems | Failed attempts ending at length limit | Fraction of failures |
|---|---:|---:|---:|
| GRPO | 2,996 / 5,896 | 1,876 | 62.6% |
| Tandem | 3,003 / 5,896 | 1,911 | 63.6% |

These counts are mechanical, not semantic review. They do not show that a longer
budget would fix the failures: errors can cause long loops, and length-limited
traces can already contain a passing answer. They motivate treating truncation
as its own outcome in any error/rescue audit.

## Next experiment to distinguish recovery ability

Use a fixed bank of independently verified base-authored errors, including errors
from ultimately successful trajectories to avoid conditioning only on failure.
Freeze the exact token prefix immediately after each erroneous step and give
both selected seniors that identical prefix with matched decoding and remaining
budget. First measure senior-only repair to isolate senior capability; then
measure repair under continued word handoff as a separate condition. Include
corrected-prefix controls to distinguish general problem difficulty from the
cost of the injected error. Native handoff replay would require explicitly
restoring/resetting sampler state under a documented common rule.

Score detection, actual correction, retention, and final correctness separately;
record new senior errors and tokens to repair. Cluster uncertainty by problem.
This is a proposed experiment, not an implemented or submitted evaluation.

For a larger automated review, first blind author and training labels during
mathematical error localization, then map exact quoted spans to the saved masks.
Allow mixed/ambiguous attribution. Compare Luna and Sol on a manually adjudicated
set containing real arithmetic errors, successful repairs, truncation loops and
answer-target mismatches; report agreement and critical misses before selecting
the cheaper judge. This pilot used manual review only: neither model's judging
quality has been calibrated, and no external API calls were made.

## Evidence and validation

[Machine-readable evidence](../results/word-handoff-case-study-20261010/evidence.json)
contains source SHA-256 hashes, saved counts, manual annotations, character
offsets, author-tagged excerpts and aggregate counts. Full word traces for the
four selected problems are copied alongside it.
[Reproduction script](../results/word-handoff-case-study-20261010/reproduce.py)
rechecks the snapshot and builds those artifacts with `uv run` from the repository
root. It requires the existing comparison export, source evaluations and tokenizer.

All selected exports were compared exactly against original saved texts and
grades and freshly decoded authorship through the existing comparison-browser
normalizer. Excerpts reconstruct their source substrings exactly. The original
evaluations, grades and ongoing code changes were left untouched. Results are
local ignored artifacts, consistent with the existing trace-browser workflow.
