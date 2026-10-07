# Training monitoring

The reproduction selects verl's `console,file` loggers, not W&B. Each scheduler
attempt writes `RUN_ROOT/{grpo,tandem}-full/metrics-JOBID-RESTART.jsonl`.
`WANDB_MODE=offline` alone does not enable a W&B logger.

## Plot current progress

Use the existing evaluation environment (NumPy and Matplotlib required). This
reads complete log lines and writes a timestamped snapshot, PNG, SVG and JSON
health summary; it does not modify training or checkpoint selection.

```bash
uv run --python "$WATCH_ENV/bin/python" --no-project --offline \
  python eval/training_progress.py --root "$RUN_ROOT" \
  --out "$RUN_ROOT/monitoring/progress"
```

To redraw a captured snapshot, replace `--root` with
`--snapshot PATH/progress.snapshot.json`. Rerun the command to refresh; the images
are snapshots, not a live dashboard. Keep interim outputs out of final results.

`capture_training_snapshot(root, out)` reads the two arms and atomically writes
the snapshot. `plot_training_progress(snapshot, out)` writes the plots and returns
the summary. Both are defined in [training_progress.py](../eval/training_progress.py).
Attempt merging uses the verifier's implementation: replayed steps replace
abandoned observations, and missing metrics remain gaps without interpolation.

- Training reward is mean binary correctness on each sampled training batch
  (16 prompts, eight rollouts each). Thin lines show raw values; thick lines show
  trailing means requiring ten consecutive observed steps. Batch difficulty
  makes these curves noisy.
- Held-out pass@4 is measured every 20 steps. GRPO uses solo rollouts and Tandem
  uses tandem rollouts; these curves do not compare their solo capabilities.
  The final Figure 2 evaluation makes that comparison separately.
- Response length and fraction reaching the 3,000-token limit help identify
  truncation. The JSON summary also checks observed metrics for nonfinite values,
  records gradient norms and Tandem's senior token fraction (target about 0.5).

## Recover a saved step whose metrics were interrupted

Preserve the relevant scheduler output before another restart overwrites it.
Then record the actual checkpoint-load evidence:

```bash
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline \
  python train/figure2_checkpoint.py --root "$RUN_ROOT/tandem-full" \
  --record-resume-step 20 --resume-log PATH/TO/PRESERVED-RESUME.log
```

This writes `resume-evidence/step-20.json`, requiring model, optimizer, RNG and
scheduler load messages for that run, a matching adjacent attempt boundary, and
intact weights. It cannot recover missing reward or validation measurements.
The selected checkpoint record lists `unobserved_metric_steps`; unexplained gaps
and missing final validation still fail verification.
