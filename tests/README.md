# Tests

CPU checks for the handoff schedule, loss mask, validation protocol, and benchmark aggregation.

## CPU

```bash
python -m unittest discover -s tests
```

| file | what it proves |
|---|---|
| `test_word_state_machine.py` | word handoff boundaries and recovery; the real sampler forward pass selects distinct senior/junior tokens consistent with its authorship mask |
| `test_senior_gate.py` | mask/padding behavior and backpropagation through the full `ppo_loss`: junior-token log-probabilities have zero direct loss gradient, while senior positions contribute |
| `test_validation_mode.py` | both validation entry points preserve rollout authorship, and baseline launchers clear inherited tandem settings |
| `test_benchmark_aggregation.py` | default evaluation covers the paper's four benchmarks, with AIME pooled before macro averages for accuracy and legibility |

The schedule and gate tests import functions from the installed forks. The validation test
executes the parameter setup from both installed agent-loop methods before Ray dispatch,
and runs the launchers with a recording stub in place of the trainer. No test loads a model
or requires a GPU or distributed group.

The sampler forward check uses deliberately different synthetic logits and greedy
decoding; it does not test GPU model forwards or weight synchronization. The loss
check tests gradients with respect to token log-probabilities. Junior tokens still
form part of the context for subsequent senior predictions.

Fork-dependent tests skip when the required package or symbol is unavailable; a skip is not a pass.
Read the skip reason: a missing package and an unpatched package look different there, and
the second is the one to worry about, since under stock vLLM the authorship mask never
arrives and training silently degrades to plain GRPO.
