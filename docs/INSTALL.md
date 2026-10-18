# Install

Use the uv-managed setup profiles in [slurm/setup.sh](../slurm/setup.sh).
Training installs the pinned editable verl/vLLM forks; paragraph evaluation uses
stock vLLM, and saved-output grading has a separate CPU-only profile.
The original `env/install.sh` is a legacy Conda installer, not the current setup path.

## Environment

Run from the repository root on a suitable Linux CPU allocation with `uv`
available. Set absolute paths for the checkout, environment and setup records:

```bash
export REPO="$PWD"
export TANDEM_ENV=/path/to/envs/tandem-training
export RUN_ROOT=/path/to/setup-records
export UV_BIN_DIR="$(dirname "$(command -v uv)")"
PROFILE=training PREPARE_FORKS=1 bash slurm/setup.sh
```

`PREPARE_FORKS=1` clones and patches the pinned sources before installation.
Use `PROFILE=inference` or `PROFILE=grading` with a separate `TANDEM_ENV`
for those runtimes. Native word-handoff evaluation needs the patched training
runtime. Pilot data preparation uses its own locked
[uv project](../experiments/shorthand/env/pyproject.toml).

See [setup inputs](WORKFLOWS.md#environment-setup) for cache and optional Apptainer
settings. The container uses the pinned CUDA 12.8.1 devel image. GPU execution
requires compatible NVIDIA hardware/drivers; CPU import checks alone do not
validate GPU execution. Size allocations under the shared GPU allocation policy,
using current site restrictions and the workload's measured requirements.

## Pins and verification

| Component | Pin/source |
|---|---|
| Python | 3.11 |
| vLLM | 0.19.1 wheel; training applies the Tandem patch to [the pinned commit](../third_party/VLLM_BASE_COMMIT.txt) |
| verl | Training only: [pinned commit](../third_party/VERL_BASE_COMMIT.txt) plus Tandem patch |
| Torch / Transformers | 2.10.0 / 5.5.4 |
| Grading | [grading requirements](../env/grading-requirements.txt), also included by inference/training setup |

[Reference freeze](../env/requirements-freeze.txt) constrains installation;
[constraints](../env/constraints.txt) protect the training stack. Setup installs
verl without its `[vllm]` extra and reuses matching precompiled vLLM extensions.
It checks imports and focused tests and writes `RUN_ROOT/PROFILE-environment.txt`.
Training checks include the patched sampler, senior loss gate and validation.
Review test skips as well as failures; see [test coverage](../tests/README.md).

## Data and launch configuration

Setup does not download model snapshots or build datasets. Run the separate
[preparation commands](WORKFLOWS.md#environment-setup), then follow the
[training protocol](FRESH_TRAINING.md) and [evaluation protocol](FIGURE2.md).

For direct training launchers, copy `train/env.sh.example` to the ignored
`train/env.sh` and set `TANDEM_ENV_BIN` to your uv environment's `bin` directory
(the template still has a legacy Conda path). Configure model, data and checkpoint
paths there. Cluster wrappers provide their own explicit environment inputs;
see [current job interfaces](WORKFLOWS.md).
