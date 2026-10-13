# Native word-handoff execution

`experiments/word_handoff/word_handoff.py --concurrency N --decode-graphs`
uses the existing vLLM native sampler with decode-only CUDA graphs. Sampling,
seeds, switching, response budgets, grading and authorship checks are unchanged.
The default remains eager execution with the previous concurrency. Each model's
KV cache grows with concurrency and maximum context; qualify memory capacity
before changing it. Both models must retain every active sequence without eviction.

`experiments/word_handoff/profile_inference.py --root CAMPAIGN --out JSON
--mode eager2|graph2|graph8 [--full]` compares fixed 512-token diagnostic draws
on one GPU, recording throughput and checked token authorship. `--full` instead
uses the real 6,000-token budget, natural stopping and executable APPS grading.
Diagnostic outputs are separate from production evaluations.

For the October 2026 APPS campaign, `experiments/coding/migrate_output_limit.py
--root CAMPAIGN --arm ARM --phase word-handoff --concurrency N` validates the
original grader, exact previous evaluation signature and dataset, backs up saved
progress, and records the new grader and execution identities before resuming.
Use only after retiring the old continuation. Previously returned grades are
unchanged by the output-limit exception fix; original responses are retained.
Different execution batching can change sampled trajectories, so the migration
does not claim bitwise equivalence for new generations.
