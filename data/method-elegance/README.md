# Method elegance dataset

374 answer-checked problems with all 1,840 supplied solution paths labeled by GPT-6.1-sol: 927 elegant and 913 non-elegant. Of these, 301 problems have at least one valid route of each style and are training eligible. The other 73 remain available for audit.

Shortness is supporting evidence, not a sufficient criterion. Median valid-route lengths are 120 words for elegant and 192 for non-elegant. Invalid or uncertain routes cannot be elegance-positive; correctness and the labeling rationale remain separate fields. Labels cover supplied paths, not every mathematically possible solution.

The archive includes 38 independent method-holdout folds, 11 with at least five test problems. Training excludes the target and declared neighboring methods from every supplied route. Do not union separate folds. Semantic leakage checks and measured senior/junior accessibility remain outstanding; these are judge annotations, not established ground-truth capability differences.

## Open locally

From the repository root:

```sh
unzip data/method-elegance/dataset.zip -d outputs/method-elegance
uv run experiments/method_audit/serve.py --data outputs/method-elegance
```

Open http://127.0.0.1:8766. The archive contains only accepted, answer-checked problems. The UI shows every supplied path and its label. `dataset.json` marks training eligibility explicitly.

See [audit interfaces and verification](../../docs/METHOD_AUDIT.md). `dataset.manifest.json` records archive and per-file SHA-256 hashes. Original source text, explicit corrections, mathematical reviews, executable verification evidence, elegance rationales, and offline KaTeX assets are included. Human audit databases and credentials are excluded.

## Sources and attribution

Questions and original solutions retain their original ownership and source terms; this release makes no new ownership or licensing claim over them. Source URLs, contest identifiers, and pinned revisions are preserved per record and in the manifest.

- [HARP](https://github.com/aadityasingh/HARP), revision `dac2734ff6443bcaf3bbdcb10f13cf21ae9729c2`.
- [HM-ReasoningBench](https://huggingface.co/datasets/jasonlwd/HM-ReasoningBench), revision `8da44ed052ae6bcdafe5bb6e1c930484e358a571`.

Questions are distributed in a compressed archive, following HARP's distribution convention. KaTeX's license is included alongside its assets.
