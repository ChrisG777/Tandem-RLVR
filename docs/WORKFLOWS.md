# Current setup and job interfaces

This is the current entrypoint map after the October 10 cleanup. Historical run
logs retain paths and commands used at the time. Original upstream files and
entrypoints remain in place; only our additions were moved or removed.
No cluster jobs or remote working directories were changed by the cleanup.
Do not update the checkout used by a running or queued job in place: a queued
script may still source the old paths. Use a separate checkout for new work.

## Environment setup

Run `bash slurm/setup.sh` on a suitable CPU allocation with these environment
variables (or pass them as `KEY=VALUE` arguments):

| Input | Meaning |
|---|---|
| `REPO`, `TANDEM_ENV`, `RUN_ROOT`, `UV_BIN_DIR` | Checkout, destination environment, setup records, uv executable directory |
| `PROFILE=inference` | Stock vLLM evaluator dependencies plus grading and plotting |
| `PROFILE=training` | Inference dependencies plus existing pinned patched verl/vLLM |
| `PROFILE=grading` | CPU-only saved-output grading dependencies |
| `HF_HOME`, `UV_CACHE_DIR` | Optional explicit shared caches |
| `APPTAINER_IMAGE`, `CONTAINER_BIND`, `APPTAINER_BIN_DIR` | Optional container image, bindings, executable location |
| `PREPARE_FORKS=1` | Training profile only: invoke the original patch installer before setup |

A missing container image is built from the existing pinned CUDA devel image.
The environment freeze is written to `RUN_ROOT/PROFILE-environment.txt`. Repeated
setup reuses the environment. Pins come from the authors' `env/requirements-freeze.txt`;
`env/grading-requirements.txt` is included by the inference requirements, so the
verification backend is not omitted from either profile. Training additionally
checks the patched imports and original mask/sampler/validation tests.

Setup does not prepare data or silently download checkpoints. Separately run:

```bash
uv run --no-project --python "$TANDEM_ENV/bin/python" python eval/download_figure2.py --out "$RUN_ROOT/base.json"
uv run --no-project --python "$TANDEM_ENV/bin/python" python data/build_deepscaler.py --out-dir "$DATA_ROOT/deepscaler"
```

For pilot generation, use the pinned project at `experiments/shorthand/env/`.
Its lockfile is unchanged. Existing environments under the former project path
are not moved or deleted; create/sync the new project environment when needed.

## Generic speculative smoke guard

```bash
uv run --no-project --python "$WATCH_ENV/bin/python" python slurm/watch-smoke.py \
  --smoke-job "$SMOKE" --full-job "$FULL" --status "$RUN_ROOT/smoke-status.json" -- \
  uv run --no-project --python "$TANDEM_ENV/bin/python" python train/figure2_checkpoint.py \
    --root "$RUN_ROOT/tandem-smoke" --arm tandem --steps 3 --mode smoke
```

The command after `--` is an argument vector, not shell text. It runs only after
the smoke exits successfully; a nonzero exit or timeout fails the guard. The
monitor cancels only the specified full job on smoke/verifier/monitor failure
and atomically records the outcome, including cancellation failures. The default
verification timeout is 300 seconds; set `--verify-timeout` before `--` to change it.
A forced SIGKILL or node loss cannot run a cancellation handler.

`slurm/watch-smoke.sbatch` accepts `REPO`, `WATCH_ENV`, `UV_BIN_DIR` and optional
cache variables as `KEY=VALUE ... --`, followed by the same watcher arguments.
Supply CPU route/resources at submission. Figure 2 artifact verification is a
separate command: `eval/check_figure2.py --results DIR`. Other experiments can
supply any verification command without changing the watcher.

`slurm/submit-training.sh` now uses this wrapper. Its original speculative
scheduling behavior remains: full jobs wait for monitor startup, not smoke success.

## Shared container and resume runtime

Launchers source `slurm/runtime.sh` after setting `JOB_ENTRYPOINT` to their
repository-relative script path. Set one progress interface:

- `RESUME_CHECKPOINT_ROOT`: trainer directory containing the commit marker,
  dataloader state, and model/optimizer/extra-state files.
- `RESUME_PROGRESS_FILE`: exact nonempty evaluation-progress file.
- `RESUME_PROGRESS_DIR`: directory of durable per-output JSON/progress files.

The host traps Slurm's pre-walltime USR1 signal and requeues the same job/array
item only with saved progress and below `MAX_RESTARTS` (default 12). The application
remains responsible for validating/restoring state. No external polling monitor
is required. Jobs using this must request `--requeue` and `--signal=B:USR1@300`.
Explicit phase selection is required for Figure 2 walltime continuation.

## Pilot submission

`bash experiments/shorthand/submit.sh` replaces the three site/task-specific
submission scripts. It takes:

- `REPO`, `RUN_ROOT`, `DATA_ROOT`, `TANDEM_ENV`, `EVAL_ENV`, `UV_BIN_DIR`.
- `PILOT_TASKS`: one or more supported task names; `PILOT_BUDGETS`: exactly two
  increasing token budgets. This campaign preserves the existing step-100 evaluation.
- `CALIBRATION_JOB`, optionally `SETUP_JOB`: successful prerequisite job IDs.
- `CPU_SBATCH_ARGS`, `TRAIN_SBATCH_ARGS`, `EVAL_SBATCH_ARGS`: explicit whitespace-
  separated flags including current account, partition, QoS, CPUs, host memory,
  walltime and GPU eligibility where applicable. No shell evaluation is performed;
  values containing embedded spaces are not supported.
- Optional cache/container variables and `PILOT_CONTEXT`, `PILOT_MAX_PROMPT`,
  `PILOT_PPO_MAX_TOKEN`, `PILOT_LENGTH_PENALTY`, `MAX_RESTARTS` are forwarded.

No route is selected automatically. Check current cluster policy and capacity
before submission. The script records each job and retains an atomic submission
lock on failure; inspect partial submissions before retrying.

For calibration, use `experiments/shorthand/slurm/shorthand-eval.sbatch` with
`PHASE=calibration`, explicit tasks/budgets, and the inference environment. Set
`CHECK_CALIBRATION=0` only for diagnostic budget probes: this preserves the old
calibration-only launcher's ability to save results that do not yet pass the gate.
The campaign submission always runs the calibration gate before training.

## Removed entrypoints

| Former entrypoint | Replacement |
|---|---|
| `slurm/continue-training.py` | Signal-based `slurm/runtime.sh` |
| `slurm/watch_figure2.py`, `slurm/watch-training.sh` | Generic smoke guard plus an explicit verification command |
| `slurm/figure2-setup.sh`, `slurm/training-setup.sh`, `slurm/shorthand-calibration-setup.sh` | `slurm/setup.sh` profiles; separate data/model preparation |
| `slurm/shorthand-calibrate.sbatch` | Pilot evaluator's calibration/probe mode |
| `slurm/submit-shorthand.sh`, `slurm/submit-engaging-pilot.sh`, `slurm/submit-concept-pilot.sh` | `experiments/shorthand/submit.sh` |

Experimental Python entrypoints moved to their corresponding subdirectory under
`experiments/{shorthand,drift,word_handoff}/`; their basenames and scientific
behavior remain unchanged. Their job scripts moved under each experiment's
`slurm/` directory. Data/results and historical run records retain their locations.
