# Figure 3: marginal token drift

[`eval/figure3.py`](../eval/figure3.py) reconstructs all four panels from the **same
three complete solo files used for Figure 2**, under `results/figure2-regraded/`.
It needs no GPU, inference, probability scoring, handoff traces, or answer grades.
The source is the supplied *Tandem RLVR.pdf*, Figure 3 (p. 6), §4.4.1 (p. 7),
and Appendix C.2 (p. 18). These are marginal emission frequencies, not the
conditional next-token cross-entropy measured by `eval/legibility.py` / Table 1.

## Interfaces

| Entry point | Contract |
|---|---|
| [`main()`](../eval/figure3.py) | Complete matching solo evaluations + pinned tokenizer → PNG/SVG, statistics JSON, reusable token-count JSONs |
| [`count_tokens(result, tokenizer)`](../eval/figure3.py) | All saved responses → integer counts indexed by tokenizer ID; no correctness filtering |
| [`summarize(counts, tokenizer, *, pseudocount, top, min_base_frequency)`](../eval/figure3.py) | Three count vectors → four panel statistics, selected token IDs, rates, and cutoff sensitivity |
| [`plot(stats, out)`](../eval/figure3.py) | Statistics → four-panel PNG/SVG |

## What each panel measures

Let `p0(t)`, `pG(t)`, `pT(t)` be each model's pooled count of token `t` divided
by its total emitted token count. Natural logs are used throughout.

- **(a)** Survival of `abs(log(ps(t)/p0(t)))`: how much token mass lies beyond
  each drift threshold. Smaller values mean fewer tokens with large marginal drift.
- **(b)** Per-token JSD contributions `0.5 * [p0 log(p0/m) + ps log(ps/m)]`,
  where `m=(p0+ps)/2`, summed within base-frequency bins `[0,1e-5)`,
  `[1e-5,1e-4)`, `[1e-4,1e-3)`, `[1e-3,1]`. These contributions sum to total
  JSD; they are **not** divergences of separately renormalized bins. Zero terms
  are evaluated exactly using SciPy `rel_entr`. Units are nats, not JSD distance.
- **(c)** `x=log(pG/p0)` versus `y=log(pT/pG)` for the selected most displaced
  GRPO tokens. `y=-x` means exact return to base; `y=0` means no further shift.
  Blue points end closer to base: `abs(x+y)<abs(x)`. Orange points do not.
  Opposite direction (`x*y<0`) is also recorded separately: overshooting can
  move in the reverse direction without ending closer. Spearman correlation
  describes these two displacements, not model accuracy.
- **(d)** Raw emission rates per million tokens for interpretable tokens ranked
  by the focal senior's ratio to base. Filled dots show base and focal senior;
  open dots show the other senior. Multipliers compare the focal senior to base.

## Explicit reconstruction choices

The paper does not specify the exact support, zero-count treatment, survival
weighting, number/eligibility of top displaced tokens, or manual interpretability
selection. The public upstream tree inspected during this reconstruction contains
no Figure 3 plotting script. The following are **our conventions**, not attributed
to the authors:

1. Pool all 1,064 problems × 32 responses per model, with no benchmark macro
   weighting or correctness filter. Include the entire saved response, including
   the final answer; omit prompts and do not add special tokens. The solo outputs
   save text rather than sampler token IDs, so re-encode with the frozen Qwen base
   tokenizer at revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Original
   decode/encode boundaries and stripped EOS cannot be recovered exactly.
2. For log ratios only, add 0.5 count to each token in the three-arm observed
   union, then renormalize each arm. JSD and displayed rates remain **unsmoothed**.
3. Panel (a) weights survival by the corresponding senior's empirical token
   frequency. Statistics also contain a uniform-over-observed-token-types variant
   because the paper's “fraction of tokens” does not settle that distinction.
4. Panel (c) selects the top 500 tokens by absolute GRPO log ratio among tokens
   whose unsmoothed base frequency is at least `1e-5`. This avoids ranking almost
   unseen tokens by zero-count artifacts. Store results for top 100/250/500/1,000
   as a cutoff sensitivity check. Neither the cutoff nor the frequency floor is
   claimed to be the authors' choice.
5. Panel (d) uses the same base-frequency floor. Require the focal senior's rate
   to exceed both base and the other senior, rank by senior/base log ratio, and
   take six distinct stripped labels per arm. A fixed filter accepts ASCII word
   fragments (2–24 characters, including spaces/apostrophes/hyphens) or 2–12
   markup characters from `#:*_=-`. Show `repr(decoded_token)` to preserve visible
   leading spaces. After inspecting the ranked candidates, manually exclude six
   subword fragments: `iad` (62944), ` comb` (3614), `inator` (17272), `lymp`
   (13842), `ics` (1211), `ence` (763). These exclusions are fixed in the script
   and recorded in JSON; they affect only panel (d), not any quantitative metric.
   This is our interpretable case study, not the authors' list, and does not cover
   every kind of interpretable mathematical token.

The supplied paper's reported 37% JSD reduction, 91% reversal, and −0.60
correlation are comparison targets, never plot inputs. Our independently trained
checkpoints also retain the training-reward limitation documented in
[the reproduction audit](REPRODUCTION_AUDIT.md): post-hoc grading does not fix
training under the original missing verifier dependencies. Grading corrections
have no effect on these token counts.

## Run and artifacts

Use the existing uv-managed `.venv` and its NumPy/Matplotlib dependencies:

```bash
uv pip install --python .venv/bin/python tokenizers==0.22.2 scipy==1.15.3
uv run --no-project python eval/figure3.py \
  --results results/figure2-regraded \
  --out results/figure3/figure3 \
  --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer.json
uv run --no-project python -m unittest discover -s tests -p test_figure3.py
```

Outputs: `results/figure3/figure3.{png,svg,json}` and
`figure3-counts-{base,grpo,tandem}.json`. Source and tokenizer SHA256 hashes
are recorded; count caches are reused only when those hashes and counting method
match. The JSON includes every plotted statistic, selected token IDs and exact
case-study strings. Change `--pseudocount`, `--top`, or `--min-base-frequency` to
examine sensitivity without recounting unchanged texts.

## Results from the completed Figure 2 evaluations

Token totals are 64,439,216 (base), 52,517,127 (GRPO), and 55,047,835 (Tandem),
from 34,048 saved responses per arm. Total JSD is 0.00355272 nats for GRPO
and 0.00301511 for Tandem, a **15.13% reduction**. The four band reductions,
from rarest to most common, are 13.22%, 22.00%, 0.62%, and 20.07%.
The survival curves cross; these runs do not reproduce the paper's claim that
Tandem's curve is lower at every threshold under our survival convention.

Among the selected 500 tokens, 83.2% end closer to base, 84.0% move in the
opposite direction, and Spearman rho is −0.548. Across top-100/250/500/1,000
cutoffs the closer fraction ranges from 80.0% to 85.0%. Changing the log-ratio
pseudocount from 0.1 to 0.5 to 1 leaves the top-500 closer fraction unchanged
and rho between −0.549 and −0.547; the additional check is saved in
`results/figure3/smoothing-sensitivity.json`. JSD does not depend on this
pseudocount. These are descriptive pooled statistics, without bootstrap intervals.

The case study does not reproduce a simple GRPO-report-markup versus
Tandem-connective-language distinction: Tandem over-emits both `Final` and
` Therefore`, while GRPO's selected tokens include ` construct`, ` minimal`,
and `Since`. Token types alone do not establish their meaning in context.
Six focused Figure 2/3 tests pass; the final PNG was visually inspected.
