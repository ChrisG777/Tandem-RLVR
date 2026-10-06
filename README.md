# Tandem RLVR: reusable shorthand

Test whether tandem RLVR preserves understandable reasoning when repeated operations and limited response budgets make shorthand useful.

**Status:** proposed pilot; task research and CSAIL setup checked, implementation and jobs awaiting the interface/design check. This branch extends [Tandem-RLVR](https://github.com/CSSLab/Tandem-RLVR) using its existing patched vLLM/verl trainer.

## Experiment

- Tasks: [Reasoning Gym](https://github.com/open-thought/reasoning-gym) matrix transformations and string rewriting, using its generators and oracle answers.
- Eight pilot runs: two tasks × solo/tandem GRPO × 256/1,024 response tokens.
- Same Qwen3-4B-Instruct-2507 initialization, data, and correctness-only reward. Tandem uses a frozen initial-model copy with 50% word-boundary authorship.
- Proposed training: 100 updates, batch 16, eight rollouts/prompt, LR 1e-6, seed 42. No supplied shorthand or readability reward.
- Disjoint generated splits and longer-chain tests; evaluate seniors alone at both budgets. Save accuracy, traces, lengths, and truncation rates. Readability requires reader comprehension tests, not token-frequency claims.

## Interfaces

| Interface | Inputs → outputs | Status |
|---|---|---|
| [Dataset builder](docs/SHORTHAND_EXPERIMENT.md#interfaces) | Pinned generator/config/seed → Parquet splits and manifest | Proposed |
| [Reward adapter](docs/SHORTHAND_EXPERIMENT.md#interfaces) | Task/response/target → binary correctness and format diagnostics | Proposed |
| [Training launchers](train/README.md) | Model/data/budget/arm → checkpoints and metrics | Existing; task overrides planned |
| [Evaluation](docs/SHORTHAND_EXPERIMENT.md#interfaces) | Checkpoint/test split/budget → solo traces and scores | Proposed |

[Experimental details and sources](docs/SHORTHAND_EXPERIMENT.md) · [Tandem architecture](docs/ARCHITECTURE.md) · [Original reproduction](docs/FRESH_TRAINING.md)
