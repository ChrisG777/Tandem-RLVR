# APPS checkpoint migration

`experiments/coding/job.sbatch train` accepts `VALIDATION_ONLY=1` with `MODE=full`
to restore the latest complete checkpoint and run its held-out validation through
verl's existing `val_before_train` / `val_only` mode. It requires the resume marker,
writes normal validation traces, logs to the console, and exits without training,
saving a checkpoint, or selecting a final model. Training metrics are untouched.
Normal training is unchanged when this variable is unset.

The October 12 GRPO migration retains checkpoint 160, the best prior validation
export (step 60), and all metric/validation history. Other exports remain on
Engaging. Step 160 was saved before preemption interrupted validation and metric
logging: recover its validation first, then resume training. Record the missing
training-metric step with `train/figure2_checkpoint.py --record-resume-step 160`
after step 161 supplies the adjacent resume boundary; never fabricate its loss.
