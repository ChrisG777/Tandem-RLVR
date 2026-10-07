"""Bounded walltime continuation for existing resumable SLURM training jobs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time


def main() -> None:
    """Monitor JOB=CHECKPOINT_ROOT pairs; requeue once near expiry after a saved checkpoint.

    Requires an existing launcher that restores optimizer/scheduler/RNG state.
    Keeps job IDs/dependencies. Never retries application failures or restarts a
    job whose existing three-restart limit is exhausted. Records actions before
    issuing them, so a monitor restart cannot repeat an unrecorded decision.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", required=True, help="JOB_ID=CHECKPOINT_ROOT")
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    jobs = dict(item.split("=", 1) for item in args.job)
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    while jobs:
        for job, root in list(jobs.items()):
            result = subprocess.run(["scontrol", "show", "job", "-o", job],
                                    text=True, capture_output=True, check=True)
            fields = dict(re.findall(r"(\w+)=([^\s]+)", result.stdout))
            status = fields["JobState"]
            if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}:
                print(f"{job}: terminal {status}", flush=True)
                del jobs[job]
                continue
            if status != "RUNNING" or job in state:
                continue
            end = datetime.fromisoformat(fields["EndTime"])
            if (end - datetime.now()).total_seconds() > 300:
                continue
            if int(fields["Restarts"]) >= 3:
                raise RuntimeError(f"{job}: launcher restart bound exhausted; manual route review required")
            marker = Path(root) / "latest_checkpointed_iteration.txt"
            step = int(marker.read_text().strip())
            checkpoint = Path(root) / f"global_step_{step}"
            actor = checkpoint / "actor"
            if (step <= 0 or not (checkpoint / "data.pt").is_file()
                    or not list(actor.glob("model_world_size_*_rank_*.pt"))
                    or not list(actor.glob("optim_world_size_*_rank_*.pt"))
                    or not list(actor.glob("extra_state_world_size_*_rank_*.pt"))):
                raise RuntimeError(f"{job}: no complete checkpoint marker at {checkpoint}")
            state[job] = {"step": step, "requested_at": datetime.now().isoformat(), "restarts_before": int(fields["Restarts"])}
            args.state.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.state.with_suffix(".partial")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(args.state)
            subprocess.run(["scontrol", "requeue", job], check=True)
            print(f"{job}: requeued for walltime continuation from checkpoint {step}", flush=True)
        if jobs:
            time.sleep(60)


if __name__ == "__main__":
    main()
