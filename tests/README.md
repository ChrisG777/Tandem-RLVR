# Tests

CPU checks for the handoff schedule, loss mask, validation protocol, and benchmark aggregation.

## CPU

```bash
python -m unittest discover -s tests
```

| file | what it proves |
|---|---|
| `test_word_state_machine.py` | the word handoff redraws the active model only at a boundary token or when `max_gap_tokens` is reached, and the recovery path reproduces the live one |
| `test_senior_gate.py` | a junior emitted position carries no gradient: the authorship mask zeroes it in the response mask `ppo_loss` uses |
| `test_validation_mode.py` | both validation entry points preserve rollout authorship, and baseline launchers clear inherited tandem settings |
| `test_benchmark_aggregation.py` | default evaluation covers the paper's four benchmarks, with AIME pooled before macro averages for accuracy and legibility |

The schedule and gate tests import functions from the installed forks. The validation test
executes the parameter setup from both installed agent-loop methods before Ray dispatch,
and runs the launchers with a recording stub in place of the trainer. No test loads a model
or requires a GPU or distributed group.

Fork-dependent tests skip when the required package or symbol is unavailable; a skip is not a pass.
Read the skip reason: a missing package and an unpatched package look different there, and
the second is the one to worry about, since under stock vLLM the authorship mask never
arrives and training silently degrades to plain GRPO.
