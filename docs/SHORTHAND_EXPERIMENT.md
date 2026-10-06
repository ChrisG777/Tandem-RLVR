# Repeated-operation shorthand pilot

Status: design proposal, not an implemented or launched experiment.

## Task research

Use [Reasoning Gym](https://github.com/open-thought/reasoning-gym), Apache-2.0, pinned to `49b07130b3fcd12f2d064bba7c43869543a0e7e7`.

| Candidate | Assessment |
|---|---|
| [Matrix transformations](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/algorithmic/manipulate_matrix.py) | Selected: recurring rotations, reflections, and mappings support reusable shorthand and operation composition. |
| [String rewriting](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/algorithmic/string_manipulation.py) | Selected: repeated priority-ordered replacement/deletion rules; oracle intermediate states allow filtering trivial cases. |
| [BIG-bench Tracking Shuffled Objects](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/tracking_shuffled_objects) | Natural permutation/state-tracking task; preserve published benchmark examples for evaluation. Less convenient than adjustable procedural generators for this pilot. |
| [DeepMind Mathematics](https://github.com/google-deepmind/mathematics_dataset) | Established compositional mathematical generators; conventional algebra is already compact, so opaque shorthand is less directly targeted. |

Reuse upstream generators; do not reimplement transformations. Upstream scoring can assign substring partial credit, which this experiment should replace with strict final-answer correctness.

## Data proposal

Per task: 4,096 train, 128 validation, 256 test, and 256 longer-chain test examples. Use separated seed/index ranges and deduplicate prompt hashes across all splits. Save generator revision, accepted indices, rejection counts, configurations, and file hashes. Both arms/budgets use identical saved data.

- Matrices: 3–4 rows/columns; 6–10 operations, with 12–16 for longer-chain tests. Use rotations, reflections, and value mappings. Disable cropping, dimension deletion, and zeroing to avoid collapsed answers. Clarify clockwise rotation and oracle reflection conventions in the prompt. Inspect no-op and unchanged-output frequencies before finalizing filters.
- Strings: input length 12–24; 8–12 rules from the upstream library. Retain 4–10 actual transformations, with 12–18 for longer-chain tests. Keep upstream cycle/termination semantics. A 20,000-example inspection confirmed thousands of eligible multi-step cases.

Check prompt lengths and baseline difficulty on a separate calibration split before finalizing. Do not tune using test examples. Both models see the same rule descriptions; no abbreviation vocabulary is provided.

## Interfaces

These are proposed signatures, not implemented APIs. Preserve existing single-completion training; no memory reset or multi-episode architecture.

- `build_dataset(task: str, out_dir: Path, seed: int = 42) -> dict`: generate verl-compatible Parquet splits and a provenance manifest, preserving oracle metadata; reject output-directory reuse and cross-split duplicates.
- `compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict[str, float]`: existing verl callback contract; extract an explicitly delimited final answer and return binary correctness plus format diagnostics. Normalize matrix whitespace while preserving shape; compare string content exactly. Malformed/missing answers score zero.
- `evaluate(model: str, data_path: Path, out_path: Path, max_tokens: int, seed: int = 17) -> dict`: GPU inference via existing vLLM utilities, explicitly solo; save responses, scores, lengths, finish reasons, and provenance.

## Pilot and interpretation

Eight runs: task × solo/tandem × 256/1,024 response tokens. Initial policy and frozen junior: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Proposed settings: 100 updates, batch 16, minibatch 8, eight rollouts, temperature 0.6, LR 1e-6, seed 42. Select the fixed final checkpoint to avoid differing solo/team validation-selection criteria.

Evaluate base and trained seniors alone at both budgets, reporting accuracy, length, and truncation. Check gradient finiteness, answer compliance, and live tandem authorship masks. One seed is exploratory. Budget pressure may produce readable notation or less reasoning rather than jargon; such outcomes do not establish interpretability improvements. Preserve traces for blinded semantic-reader evaluation. Reader transfer and definition-rescue interventions need concrete designs after the pilot exposes the representations being learned.

## CSAIL preparation

Isolated checkout: `/data/scratch/chrisge/Tandem-RLVR-shorthand`. Reuse the existing uv-managed environment `/data/vision/torralba/u/chrisge/tandem-rlvr/train-venv` without changing its packages. Use new output directories. Solo needs one policy GPU; tandem adds a frozen-junior GPU. Use compatible shared 80 GB+ GPUs, `vision-torralba`, `shared-if-available`, requeue, and six-hour limits. Existing smoke accounting reached approximately 144 GiB host RAM, so size requests from that evidence. Record commit, data hashes, configuration, and job IDs.

No experiment jobs have been submitted yet.
