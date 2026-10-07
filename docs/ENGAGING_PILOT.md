# Independent Engaging calibration race

The first CSAIL calibration (2565219) failed: at 256 tokens both tasks had zero
correct/complete answers. At 1,024 tokens matrix accuracy was 3.9%, with only
4.7% complete answers; string accuracy/completion were both zero. Dependent
pilot jobs were automatically cancelled before training. Existing Figure 2
training jobs are separate and are not cancelled by this race.

Probe 2,048 and 3,072 generated tokens on the existing 64-example calibration
split per task, four samples each, keeping the base revision, data, prompt,
temperature 0.6, top-p 1, and evaluation seed 17 fixed. The 4,096-token inference
context remains unchanged. Each cluster writes to its own new campaign directory.
These are diagnostic calibration results, not extra independent scientific seeds.
Do not launch training until a usable reward signal and completion rate are
established; the original calibration gate remains intact.

The [CPU setup](../slurm/shorthand-calibration-setup.sh) installs the pinned
evaluation environment using uv, downloads the pinned base, and verifies the
saved datasets. The [GPU array](../slurm/shorthand-calibrate.sbatch) has exactly
two elements: matrix (0) and string (1), one GPU each. It uses stock vLLM 0.19.1
for solo inference; no Tandem patch or training runtime is required at this stage.

Required site inputs: `REPO`, `RUN_ROOT`, `DATA_ROOT`, `TANDEM_ENV`, `HF_HOME`,
`UV_CACHE_DIR`, `UV_BIN_DIR`. On Engaging use `mit_general`, CPU partition
`mit_normal`, GPU partition `mit_normal_gpu`, and an `afterok` dependency on
setup. Calibration needs one 40 GB-or-larger GPU, four CPUs, 32 GiB RAM and one
hour per element; either documented L40S or H200 fits the 4B inference model.
No array throttle, duplicate third GPU worker, or automatic cross-cluster
cancellation is used. Code is cloned/pulled through GitHub HTTPS on Engaging.
