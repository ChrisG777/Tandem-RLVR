"""Guard any speculative Slurm job with a smoke job and a verification command."""
import argparse
import json
import re
import signal
import subprocess
import time
from pathlib import Path


def main() -> None:
    """Monitor Slurm and execute argv after smoke success; never use a shell.

    Cancel only --full-job if the smoke, verifier, or monitor fails. Write status
    atomically. Transient scheduler lookup failures are tolerated five times.
    Verification must finish within --verify-timeout seconds (default 300).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-job", required=True, type=job_id)
    parser.add_argument("--full-job", required=True, type=job_id)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--verify-timeout", type=float, default=300)
    parser.add_argument("verify", nargs=argparse.REMAINDER, help="-- executable arguments...")
    args = parser.parse_args()
    command = args.verify[1:] if args.verify[:1] == ["--"] else args.verify
    if not command or args.smoke_job == args.full_job or args.verify_timeout <= 0:
        parser.error("Require a verification command, distinct jobs, and a positive timeout")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    passed = False
    record = {"smoke_job": args.smoke_job, "full_job": args.full_job,
              "passed": False, "status": "monitoring"}
    try:
        save_status(args.status, record)
        misses = 0
        while True:
            proc = subprocess.run(
                ["sacct", "-X", "-n", "-P", "-j", str(args.smoke_job),
                 "--format=JobID,State,ExitCode"],
                capture_output=True, text=True, timeout=30,
            )
            rows = [line.split("|") for line in proc.stdout.splitlines()]
            row = next((r for r in rows if r[0] == str(args.smoke_job)), None)
            if proc.returncode or row is None:
                misses += 1
                if misses >= 5:
                    raise RuntimeError("Smoke state unavailable for five consecutive checks")
            else:
                misses = 0
                state, code = row[1:3]
                print(f"smoke {args.smoke_job}: {state} {code}", flush=True)
                if state == "COMPLETED":
                    if code != "0:0":
                        raise RuntimeError(f"Smoke exit code {code}")
                    subprocess.run(command, check=True, timeout=args.verify_timeout)
                    passed = True
                    record.update(passed=True, status="smoke and artifacts passed")
                    break
                if state not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING",
                                 "REQUEUED", "REQUEUE_FED", "RESIZING", "SUSPENDED"}:
                    raise RuntimeError(f"Smoke ended with {state}: {code}")
            time.sleep(30)
    except BaseException as exc:
        record["status"] = str(exc)
        raise
    finally:
        try:
            if not passed:
                proc = subprocess.run(["scancel", args.full_job], timeout=30)
                record["cancel_returncode"] = proc.returncode
                if proc.returncode:
                    raise RuntimeError(f"Unable to cancel guarded job {args.full_job}: {proc.returncode}")
        except BaseException as exc:
            record["cancel_error"] = str(exc)
            raise
        finally:
            save_status(args.status, record)


def job_id(value: str) -> str:
    """Accept a numeric Slurm job or array-task ID, never scheduler options."""
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Expected a job ID or jobID_arrayTaskID")
    return value


def interrupted(signum: int, frame: object) -> None:
    """Convert scheduler termination into cancellation of the guarded full job."""
    raise RuntimeError(f"Monitor interrupted by signal {signum}")


def save_status(path: Path, record: dict) -> None:
    """Atomically publish monitoring state; full results remain provisional until passed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
