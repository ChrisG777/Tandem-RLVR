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

## Submitted 2026-10-07 UTC

| Cluster | Job | Work |
|---|---|---|
| Engaging | 25111321 | CPU environment/model setup; four CPUs, 16 GiB, 90 minutes |
| Engaging | 25111322_0–1 | Matrix/string calibration, dependent on successful setup |
| CSAIL | 2581354_0–1 | Matching calibration using the existing training environment |

Engaging checkout: `/orcd/scratch/orcd/013/cge7/Tandem-RLVR`, branch `main`.
Campaign: `/orcd/scratch/orcd/013/cge7/tandem-rlvr/calibration-race-20261007`.
CSAIL checkout: `/data/scratch/chrisge/Tandem-RLVR-shorthand`, branch `main`.
Campaign: `/data/vision/torralba/u/chrisge/tandem-rlvr/shorthand-calibration-race-20261007`.
Submitted implementation: `f70d51e`. Engaging normalized the generic GPU request
to one L40S per array element; each has four CPUs, 32 GiB and a one-hour limit.
Training has not been resubmitted: selecting viable budgets requires these
calibration results. No CSAIL jobs were cancelled by this submission.

## Engaging runtime compatibility recovery

Setup 25111321 failed after 59 seconds on Rocky 8 (glibc 2.28). The pinned
vLLM 0.19.1 Linux wheel requires glibc 2.31; uv fell back to a source build,
which failed with `CUDA_HOME is not set` on the CPU setup node. Array 25111322
was automatically cancelled by its failed dependency and consumed no GPU time.

Engaging setup and calibration now accept `APPTAINER_IMAGE` and `CONTAINER_BIND`.
Setup builds an official NVIDIA CUDA 12.8.1 runtime / Ubuntu 22.04 image pinned
to its amd64 OCI digest, then installs the same pinned packages inside that
userspace. GPU calibration uses the same image with Apptainer `--nv` to expose
the allocated GPU and host driver. The environment, datasets, model cache and
outputs are mounted from scratch. Setup uses `--only-binary vllm` so an
incompatible platform fails explicitly instead of starting a source build.
CSAIL's native runtime is unchanged.

Replacement submissions: CPU setup **25112692**, then GPU calibration array
**25112693_0–1** with a successful-setup dependency. Both were pending at
submission. Implementation commit: `4c15a46`. New campaign:
`/orcd/scratch/orcd/013/cge7/tandem-rlvr/calibration-container-20261007`.
The earlier failed environment/campaign is preserved. Container execution is
not yet verified; these job IDs supersede 25111321/25111322 on Engaging only.

Subsequent check: 25112692 failed on node1602 after one second with
`apptainer: command not found`; 25112693 was automatically cancelled before GPU
execution. The shared Apptainer 1.4.2 installation is now selected explicitly.
Recovery resources are being used first for [reproduction base evaluation](FIGURE2_RUN.md),
not another Engaging calibration submission. CSAIL calibration 2581354 remains
queued (a temporary hold was released at the user's request).
