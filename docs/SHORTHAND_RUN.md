# Solo shorthand pilot: 2026-10-06

Update (2026-10-07 UTC): calibration 2565219 failed its scientific gate because
the short budgets truncated almost all answers. Dependent pilot training and
evaluation were automatically cancelled before starting. No policy was trained
by this pilot and there is no conclusion about jargon emergence. See the
[larger-budget calibration history](ENGAGING_PILOT.md). The table below records the
original submissions.

| Job | ID | Purpose |
|---|---|---|
| Calibration | 2565219 | Base model on 64 separate examples/task at both budgets; gates training |
| Solo training array | 2565234_0–3 | Matrix 256, matrix 1,024, string 256, string 1,024; 100 updates each |
| Base evaluation | 2565235 | Untrained model on ordinary/longer-chain tests, both budgets |
| Trained evaluation array | 2565236_0–3 | Each final checkpoint evaluated at both budgets after its training finishes |
| Review preparation | 2565237 | CPU job creating paired metrics, blinded traces, and separate authorship key |

- Implementation submitted at commit `322e7bc`; subsequent documentation updates do not alter training.
- Checkout: `/data/scratch/chrisge/Tandem-RLVR-shorthand`.
- Campaign: `/data/vision/torralba/u/chrisge/tandem-rlvr/shorthand-solo-20261006`.
- Data: versioned `data/shorthand/` under the checkout.
- Runtime: existing uv-managed `/data/vision/torralba/u/chrisge/tandem-rlvr/train-venv`.
- Calibration initially requested two hours and was updated in place to one hour to improve backfilling. Training allows six hours; full evaluation two hours.
- Resource/dependency verification: training has one GPU, six CPUs, 160 GiB RAM, 80 GB+ GPU constraints, and successful-calibration dependency. Evaluation uses corresponding successful training-array elements. Review requires all base/trained evaluations to succeed.
- Local validation: four focused reward/data-integrity tests and three existing checkpoint-integrity tests passed; Python compilation, shell syntax, and diff checks passed. GPU execution is not yet verified for the new tasks.

Outputs are `calibration/`, `train/<task>-b<budget>/`, `eval/`, and `review/` under the campaign directory. Inspect `review/metrics.json` and `review/blinded_pairs.jsonl` before consulting `review/answer_key.json`. Calibration failure blocks training; failed training blocks its evaluation. No unrelated jobs or environments were modified. No tandem runs are part of this pilot.
