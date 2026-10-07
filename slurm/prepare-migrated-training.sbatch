#!/usr/bin/env bash
# CPU gate: wait for successful rsync, preserve the imported best weights, verify history.
set -euo pipefail
if [ "$#" -gt 0 ]; then export "$@"; fi
cd "${REPO:?}"
if [ "${TANDEM_CONTAINER_ACTIVE:-0}" != 1 ]; then
    for ((i=0; i<360; i++)); do
        [ ! -f "$RUN_ROOT/TRANSFER_COMPLETE" ] || break
        sleep 30
    done
    test -f "$RUN_ROOT/TRANSFER_COMPLETE"
    for item in grpo:120 tandem:80; do
        ARM=${item%:*}; STEP=${item#*:}
        HF="$RUN_ROOT/$ARM-full/hf/global_step_$STEP"
        if [ -L "$HF" ]; then
            rm "$HF"
            cp -al "$RUN_ROOT/$ARM-full/global_step_$STEP/actor/huggingface" "$HF"
        fi
    done
    export TANDEM_CONTAINER_ACTIVE=1
    exec /orcd/software/core/001/pkg/apptainer/1.4.2/bin/apptainer exec \
        --bind "${CONTAINER_BIND:?}" "$APPTAINER_IMAGE" bash "$REPO/slurm/prepare-migrated-training.sbatch"
fi
export PATH="${UV_BIN_DIR:?}:$PATH" HF_HUB_OFFLINE=1
for item in grpo:120 tandem:80; do
    ARM=${item%:*}; STEP=${item#*:}
    uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python \
        train/figure2_checkpoint.py --root "$RUN_ROOT/$ARM-full" --arm "$ARM" --steps "$STEP" --mode full
done
uv run --python "$TANDEM_ENV/bin/python" --no-project --offline python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import torch
root = Path(os.environ["RUN_ROOT"])
data = root / "data/deepscaler"
manifest = json.loads((data / "manifest.json").read_text())
for name, digest in manifest["sha256"].items():
    if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Migrated data hash mismatch: {name}")
for arm, step in (("grpo", 120), ("tandem", 80)):
    run = root / f"{arm}-full"
    if json.loads((run / "data-manifest.json").read_text()) != manifest:
        raise ValueError("Migrated training data manifest differs")
    checkpoint = run / f"global_step_{step}"
    for name in ("model", "optim", "extra_state"):
        path = checkpoint / "actor" / f"{name}_world_size_1_rank_0.pt"
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing resume state: {path}")
    if not (checkpoint / "data.pt").stat().st_size:
        raise ValueError("Missing dataloader state")
    # These are trusted, locally produced training artifacts, not downloaded code.
    data_state = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)
    extra = torch.load(checkpoint / "actor/extra_state_world_size_1_rank_0.pt",
                       map_location="cpu", weights_only=False)
    if not isinstance(data_state, dict) or not data_state or not extra.get("rng") or not extra.get("lr_scheduler"):
        raise ValueError("Incomplete dataloader/RNG/scheduler resume state")
print("MIGRATION VERIFIED: source history, retained best weights, resume files and data manifest")
PY
