# Local trace comparison

The browser reads saved evaluations; it does not run models or change grades.
UI code lives in `ui/comparison/` (HTML/CSS/JavaScript). Data validation,
joining, authorship decoding, export, and the localhost launcher live in
[`eval/comparison_browser.py`](../eval/comparison_browser.py).

## Run

From the repository root:

```sh
uv run eval/comparison_browser.py --skip-build --serve
```

Open <http://127.0.0.1:8765>. This serves the existing export on loopback only.
To rebuild from local evaluation files and serve:

```sh
uv run eval/comparison_browser.py --serve
```

Dependencies are pinned in the script's uv metadata. The saved results must be
present under `results/figure2-regraded/`; generated files go under
`outputs/comparison-browser/`. Both directories are ignored by git. Override
with `--results`, `--out`, `--tokenizer`, and `--port`.
The tokenizer is the pinned base model's `tokenizer.json`. For a fresh checkout:

```sh
mkdir -p outputs/comparison-browser-assets
curl -fL -o outputs/comparison-browser-assets/tokenizer.json \
  https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer.json
uv run eval/comparison_browser.py --serve
```

No network is required after dependencies, tokenizer and results are available.
Close the server with Ctrl-C. Refresh after rebuilding an export.

## Interfaces

| Interface | Contract |
|---|---|
| `build_browser(results: Path, out: Path, tokenizer_path: Path) -> dict` | Read the five configured evaluations, validate benchmark hashes and record alignment, export per-problem/condition samples and an index; return the index |
| `normalize_sample(row: dict, index: int, tokenizer) -> dict` | Preserve exact text and saved correctness; decode and validate optional token-authorship spans; fail on invalid masks |
| `index.json` | Condition provenance, protocol caveat, problem prompts/reference answers and per-condition observed success counts |
| `problems/{problem-id}-{condition-id}.json` | All saved samples for one condition/problem, requested only when displayed |

The loader reuses `common.problems_from_result`, including its global-index
semantics, then joins by benchmark and dataset row. It checks recorded dataset
hashes before presenting prompts. Completed `word-handoff.json` files take
precedence over `word-interim.json` snapshots. Export remains a snapshot until
rebuilt. UI labels describe the selected step-160 GRPO and step-180 Tandem campaign.
Do not point this configuration at different checkpoints without updating labels.

## Review behavior

Default coverage is the intersection of all five conditions. “All problems”
includes missing-condition notices. Success-rate filters compare each condition's
fraction of saved correct samples; these filters are exploratory, not significance
tests. Sample indices are independent draws, not matched random seeds.

Word handoff means each trained senior collaborates with the frozen base; it does
not mean the two trained models collaborate with each other. Green spans are
senior-authored, purple spans base-authored. Rare Unicode characters assembled
from tokens by both authors are marked amber. A decoding mismatch disables coloring
for that sample and displays a warning; source text is always preserved. Solo
artifacts lack token IDs and finish reasons, so the browser does not invent them.

Text, including LaTeX, is rendered verbatim through DOM text nodes. Model output
cannot inject HTML. No CDN assets, telemetry, or external scripts are used.
Problem selection is saved in the URL fragment for bookmarking. Each panel has
its own sample selector and scroll area; checkboxes select visible conditions.
