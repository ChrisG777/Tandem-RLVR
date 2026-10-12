# Method-pair curation and audit

Method Atlas is a local review tool for identifying tasks with a structural
solution and an elementary alternative. Source solutions, method proposals,
mathematical audits, and empirical senior/junior capability evidence remain
separate. No training or model inference is part of this tool.

## Run

```sh
uv run experiments/method_audit/curate.py
uv run experiments/method_audit/serve.py
```

Open <http://127.0.0.1:8766>. The first command downloads pinned public releases
and local KaTeX 0.16.22 assets into `outputs/method-audit/`. Subsequent launches
need only the second command, with no network access. Restart the server after
rebuilding the catalog; each running server uses one immutable catalog snapshot. Both commands accept
`--out` / `--data`, respectively; the server also accepts `--port`.

The initial screening constructs a shortlist using: a supplied answer, at least two human
solution texts, differing lexical method evidence, and no detected missing
choices or diagram markup. This is a screening heuristic, **not a guarantee of
distinct correct strategies or junior accessibility**. Uncheck “Screened shortlist
only” to browse the broader pool. Search indexes titles, question previews, and
method names; source/status/method/partition filters compose. J/K browse and
Ctrl/Command+Enter saves. Navigating to another problem saves pending changes;
closing the page with unsaved edits produces a browser warning.

## Sources and initial snapshot

The initial pool has 7,891 normalized-question records; 7,313 have multiple
retained routes and 2,808 have standalone source answers. The screened queue has
819 candidates covering 56 of 57 proposed method families. These are candidate
counts, not verified-pair counts. Initial curation marks no pairs accepted; the later Codex review pass supplies separate verdicts.

| Source | Pinned revision | Use and limitation |
| --- | --- | --- |
| [HARP](https://github.com/aadityasingh/HARP) | `dac2734ff6443bcaf3bbdcb10f13cf21ae9729c2` | Full human solutions; alternate texts may duplicate a method or contain incomplete arguments |
| [HM-ReasoningBench](https://huggingface.co/datasets/jasonlwd/HM-ReasoningBench) | `8da44ed052ae6bcdafe5bb6e1c930484e358a571` | Human/model pairs; many proof problems lack standalone answers; model correctness is not assumed |
| [ProofRank](https://huggingface.co/datasets/INSAIT-Institute/ProofRank) | `9a6ec576e2de7929d9b72cc7964eccf6577adb35` | Method summaries with answers; summaries are not full proofs |
| [Braingle](https://github.com/stephenxia1/brainteasers) | `aa242baa9bae20c8841e1cc8eacd442c4e281d31` | Human solutions and source annotations; second routes still needed |

`catalog.json` records exact download URLs and SHA-256 hashes. Raw source files
remain local under ignored `outputs/`; source ownership and licenses are not
replaced by this project. SenseMath is intentionally not imported into the
initial pool because inspected shortcut annotations need additional validation.

Question normalization removes whitespace, normalizes case, and strips LaTeX
left/right delimiters for duplicate detection. Matching questions merge sources
and distinct solution texts. This is **not** semantic or near-duplicate removal.
Method labels use transparent regex evidence, viewable by hovering on chips.
They can miss unnamed techniques or match irrelevant mentions; human review is
required. The named families are a proposed taxonomy, not empirically established
clusters. Three research seeds are prioritized with specific curation notes;
other candidates are ranked mechanically by differences in method mentions.

## Audit workflow

1. Read the question and original routes. Choose any two routes for comparison.
   Left/right order does not imply an elegant/elementary classification.
2. Assign the elegant and elementary route roles in the review panel. Confirm
   mathematical validity and substantive method difference separately. Record
   missing assumptions or corrections; original source texts remain unchanged.
3. Review method labels on **all** routes, including unused alternatives. Add
   methods implied by the question. Confirm label completeness only after review.
4. Supply a standalone answer if missing. Replace a summary with a complete
   derivation before using it as one of the accepted routes.
5. Accept, reject, or mark needs work; save. A separate capability field remains
   “Not tested” unless an actual experiment supplies evidence (record it in notes).
6. Export approved pairs or back up the entire audit state as JSON.

Audits live in `outputs/method-audit/audits.sqlite3`; rebuilds never overwrite
this database. SQLite transactions and revision checks reject stale simultaneous
edits. A changed source question, answer, route, or method proposal invalidates
previous correctness/method confirmations until the record is reviewed again.
Backups include all nonempty human audit records, machine reviews with loaded
execution evidence, and method partition settings; retain
the catalog/source hashes with them. The SQLite file is the authoritative local
store. No background sync or external service is used.

## Whole-method evaluation holdouts

The initial proposal assigns **15 labels to test** and **9 to
validation**, leaving **33 for train**. These label assignments are provisional,
not a claim of 57 independent strategies; see the [family audit](METHOD_FAMILIES_AUDIT.md). The UI’s “Methods & holdouts” table makes
these assignments editable and shows coverage. Initial screened-candidate
assignment is 432 train / 56 validation / 331 test; no pair is export-ready until
accepted and fully audited. Narrow labels added during pair review appear in the UI.

For every problem, take the union of audited method labels over **all retained
routes**, plus additional methods required by the question. Test takes precedence
over validation, which takes precedence over train. A held-out alternate route
therefore excludes the whole problem from train, even if a different route was
selected for the audit. A route with no labels blocks export rather than silently
entering training. Summaries and missing standalone answers also block export.

This establishes disjointness of the *recorded labels*, not a proof of semantic
method separation. Before training, check near duplicates, related formulations,
unnamed techniques, and parent/child method overlap. In particular, broad methods
such as symmetry may overlap narrower techniques. Freeze this taxonomy and
partition assignment before any training, and preserve the exported manifest.
These datasets may already occur in model pretraining; “held out” refers to our
Tandem training, not all prior exposure.

## Interfaces

| Interface | Contract |
| --- | --- |
| [`Problem`, `SolutionRoute`](../experiments/method_audit/curate.py) | Immutable source content, provenance, and explicitly provisional method evidence |
| [`Audit`](../experiments/method_audit/serve.py) | User verdict, role choices, correctness/distinctness checks, notes, and capability status; method/text overrides stored alongside |
| [`build_catalog(out: Path) -> dict`](../experiments/method_audit/curate.py) | Download pinned data if absent; merge exact normalized questions; emit catalog, source hashes and local math assets; preserve audits |
| [`AuditStore.save_audit(pid: str, payload: dict) -> dict`](../experiments/method_audit/serve.py) | Validate route/method references; transactionally save a complete audit only at the expected revision |
| [`AuditStore.export() -> dict`](../experiments/method_audit/serve.py) | Return accepted, fully reviewed pairs in train/validation/test partitions; report accepted records blocked by missing evidence |
| [`methods.json`](../experiments/method_audit/methods.json) | Family ID, display label, domain, default partition, and lexical matching expression |

The server binds loopback only. Static requests are limited to UI/math assets.
Writes require JSON and same-origin requests, and use no executable source text.
Math rendering uses local KaTeX with trust disabled. Python’s standard library
provides HTTP and SQLite; PyArrow is used only to read the published Parquet data.

Export records preserve `{problem, audit, methods, split, capability_evidence}`.
An audit’s `route_text` and `answer` override source text only when nonempty;
originals remain available. The export includes the catalog SHA-256 and method
partition revision. An empty export is expected before any pair passes review.

## Validation

```sh
uv run --no-project python -m unittest tests.test_method_audit -v
node --check ui/method-audit/app.js
```

Tests cover persistence, stale writes, unused-route holdout leakage, unknown
labels, missing answers, summary replacement, changed-source invalidation, and
exact duplicate merging. UI inspection checks local math rendering, filters,
method partitions, and persistence through a reload.

## Codex pair reviews

Completed snapshot (2026-10-11): **819/819 screened questions and 3,486/3,486
supplied routes reviewed**. The result is **374 accepted pairs, 319 rejected,
126 needs work**. All 374 accepted pairs have current, independently executed
passing answer checks and exactly two retained reference routes. Review schema,
source fingerprints, complete route coverage and canonical method IDs passed
the final record audit; all 23 focused software tests passed.

The accepted pairs use 61 shortcut labels, including broad tools. Only **38
defined strategy targets** receive conservative method-holdout folds, and only
**14 folds contain at least five test pairs**. These counts do not establish
independence of every family or a measured stronger/weaker-model capability gap.
The [coverage snapshot](../outputs/method-audit/machine-review/summary.json),
[checked pairs](../outputs/method-audit/machine-review/checked-pairs.json), and
[holdout folds](../outputs/method-audit/machine-review/holdout-folds.json) preserve
the final data and source/taxonomy hashes. They are local generated artifacts.

Illustrative retained pairs include Möbius inversion versus recursive divisor
equations (`8df90e9767c72418b255`), hook lengths versus center-value casework
(`38a0bacf0edb5a9e7da2`), and small modular residues plus a bound versus exact
fifth-power arithmetic (`fee784be961127321aee`). Each is a mathematical review
candidate for subsequent model calibration.

The second pass reads each of the 819 screened HARP questions and **all** its
supplied routes with a native Codex GPT 6.1 Sol reviewer. This is model-assisted
mathematical review, not a human audit or a measured junior capability result.
The CLI backend rejected the requested model on this account; the reviews use
native Codex reviewers instead. Completed records are checkpointed individually;
unseen questions are never assigned a verdict by a heuristic.

Each review records route-by-route correctness and manually assigned methods,
the two selected route IDs, the mechanism of the shortcut, why the longer route
is naturally discoverable, concerns, and an executable answer check where a pair
is accepted. Source repairs are explicit `route_text` overrides with
correction notes; the original is retained. Missing diagrams, unjustified bounds,
answer-choice tricks, incomplete proofs and nearly identical routes are grounds
for rejection or `needs_work`. Exactly two **retained reference routes** is an
export invariant, not a claim that only two mathematical solutions exist.

The UI defaults to checked AI pairs and shows the running reviewed/accepted/check
counts. Expand a pair's evidence to see all route assessments and checker output.
The AI verdict filter also exposes rejected, needs-work and unreviewed records.
Human edits and verdicts remain in the original SQLite database and are never
silently populated by machine review.

### All-path elegance labels (version 2)

The follow-up judge reads **all 1,840 supplied paths of the 374 accepted
problems**, using explicit corrected text where present. It labels each path
`elegant: true/false`, with a specific reason, confidence, and the role of shortness
(`supporting`, `neutral`, or `misleading`). Prior selected roles are not supplied
to this pass. Several paths can be elegant; no exactly-one-positive rule is imposed.

An elegant path must be complete and have a natural, findable reduction in
mathematical work or branching, or be a genuinely concise direct solution.
Shortness is evidence, not a sufficient criterion: compressed large computations,
unsupported claims, omitted cases and bare answers do not qualify. Invalid or
uncertain references cannot be elegance-positive; their reasons distinguish the
proof defect from style. These remain model judgments, not measured accessibility.

The UI's **All solution paths** panel lists every label and rationale, and lets
you place any path in comparison pane A or B. The **Export labeled paths** button
exports all paths, including explicit invalid/uncertain flags. Export waits until
every accepted problem has a complete, current judgment. The legacy
`/api/machine-export` still provides the original two-reference view.

| Interface | Contract |
| --- | --- |
| [`judgment_input(problem, review)`](../experiments/method_audit/elegance.py) | Effective question/routes/correctness → fingerprinted judge input, without prior pair roles |
| [`load_judgment(root, problem, review)`](../experiments/method_audit/elegance.py) | Load and validate complete labels; absent returns `None`, stale or malformed raises |
| [`export_dataset(root, problems, reviews)`](../experiments/method_audit/elegance.py) | Checked reviews from `AuditStore.machine_reviews()` → all-path dataset; incomplete annotation blocks export |
| [`write_package(data, destination)`](../experiments/method_audit/package.py) | Validated dataset → deterministic portable ZIP and hash manifest; no upload or human-audit inclusion |

Training eligibility additionally requires at least one valid elegant and one
valid non-elegant route. All other problems remain in the audit release with
explicit exclusion reasons. Holdout construction excludes all related labels
on every supplied path. A test problem's previously identified decisive route
must also be confirmed elegant by this second pass.

Package after all judgments and answer checks pass:

```sh
uv run experiments/method_audit/package.py --out data/method-elegance/dataset.zip
```

The ZIP includes the all-path `dataset.json`, original catalog subset,
mathematical reviews, executable-check evidence, elegance reviews, method
definitions, conservative independent `holdout-folds.json`, and pinned KaTeX
assets/license for offline browsing. `manifest.json` hashes every included file.
Human SQLite audits and credentials are never packaged. Source questions remain
compressed following HARP's distribution convention; provenance and original
source ownership are preserved.

To browse the committed dataset on another checkout:

```sh
unzip data/method-elegance/dataset.zip -d outputs/method-elegance
uv run experiments/method_audit/serve.py --data outputs/method-elegance
```

| Interface | Contract |
| --- | --- |
| [`review_schema.json`](../experiments/method_audit/review_schema.json) | JSON Schema for one durable reviewer record: all-route assessments, selected pair, source fingerprint, explicit corrections, provenance and concerns |
| [`load_reviews`](../experiments/method_audit/review.py) | Load durable reviews and hash-matched independent execution evidence; ignore self-asserted execution success |
| [`verify_reviews`](../experiments/method_audit/verify.py) | Execute accepted-pair checkers in a local sandbox and write separate evidence sidecars; never overwrite reviews or human audits |
| [`machine_export`](../experiments/method_audit/review.py) | Export exactly two valid selected routes only after current-source review and executable answer corroboration; use all audited route labels for provisional partitions |

```sh
uv run experiments/method_audit/verify.py
# Explicitly rerun checks, including previously failed ones:
uv run experiments/method_audit/verify.py --retry
```

The verifier uses a macOS sandbox, restricted arithmetic imports and a per-check
time limit. SymPy 1.14.0 and its pinned ANTLR runtime support exact numeric LaTeX
comparison; rounded decimals do not certify algebraic answers. SymPy's
[LaTeX parsing interface](https://docs.sympy.org/latest/modules/parsing.html)
is used instead of evaluating source text as Python. Passing execution corroborates
the requested answer; mathematical validity of the written routes remains a
separate reviewer judgment. A stale review hash, failed check, unknown method or
unassessed route blocks the machine export.

Machine records live under `outputs/method-audit/machine-review/reviews/`, with
separate `verification/` evidence. The UI exposes `/api/machine-export`; its
partitions remain explicitly **provisional** pending semantic holdout checks.
The [family audit](METHOD_FAMILIES_AUDIT.md) distinguishes genuine mechanisms,
generic tools and overlapping representations. Manual refinements—including mass points, hook lengths, Möbius inversion,
Farey bounds, Burnside averaging and isogonal cevians—were added after inspecting
actual pairs. They have no keyword matcher, so adding
them does not rewrite original route labels or invalidate existing pair reviews.
The vocabulary size is not a count of established independent families.

Additional focused checks:

```sh
uv run --with sympy==1.14.0 --with antlr4-python3-runtime==4.11.1 python -m unittest tests.test_method_machine_reviews tests.test_method_verification tests.test_method_audit -v
```

## Coverage snapshot and conservative evaluation folds

```sh
uv run experiments/method_audit/report.py
```

[`snapshot_reviews(data)`](../experiments/method_audit/report.py) writes
`machine-review/summary.json`, `checked-pairs.json`, and `holdout-folds.json`.
The summary records missing IDs, verdict counts, accepted-but-blocked pairs, and
confirmed shortcut-label coverage with definitions. Run after verification to
freeze the current evidence; while review is in progress, these are partial snapshots.

[`build_holdout_folds(pairs, taxonomy)`](../experiments/method_audit/report.py)
creates separate leave-one-method-out experiments for defined strategy families.
A test item uses the target as its elegant mechanism. A training item contains
none of the target or its immediate overlap-neighborhood labels in **any** reviewed
route, including unused alternatives. Broad tools and mixed/topic labels are not
eligible targets. Manual refinements inherit their declared overlap neighborhoods.
These folds are independent experiments: do not union their training and test lists.

This is stricter than the editable UI's provisional label partitions. It guarantees
recorded-method exclusion, while semantic near-duplicate screening and empirical
junior calibration remain separate requirements. Sparse folds are exposed with
counts rather than presented as adequately powered evaluations.

Published all-path snapshot: **374 problems / 1,840 routes**, with **927 elegant / 913 non-elegant** labels and **301 training-eligible problems**. The release has **38** independent strategy holdouts, **11** with at least five test items. See the [dataset card](../data/method-elegance/README.md) for provenance and portability.
