"""Cancel a speculative full job if its smoke job or output verification fails."""
import argparse
import json
import signal
import subprocess
import time
from pathlib import Path


def main() -> None:
    """Poll a smoke job, verify five result files, and record the full job's gate status.

    Run in an independent CPU allocation. Exceptions and termination cancel only
    --full-job; successful verification leaves it running. No jobs are retried.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-job", required=True, type=int)
    parser.add_argument("--full-job", required=True, type=int)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args()
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
                 "--format=JobIDRaw,State,ExitCode"],
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
                    verify_outputs(args.results)
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
        if not passed:
            proc = subprocess.run(["scancel", str(args.full_job)], timeout=30)
            record["cancel_returncode"] = proc.returncode
        save_status(args.status, record)


def interrupted(signum: int, frame: object) -> None:
    """Convert scheduler termination into cancellation of the guarded full job."""
    raise RuntimeError(f"Monitor interrupted by signal {signum}")


def verify_outputs(root: Path) -> None:
    """Require all five smoke outputs, complete chains, and two binary grades/problem."""
    for arm, phase in (("base", "solo"), ("grpo", "solo"), ("tandem", "solo"),
                       ("grpo", "handoff"), ("tandem", "handoff")):
        path = root / arm / f"{phase}.json"
        result = json.loads(path.read_text())
        if (result["phase"] != phase or result["n"] != 2 or result["limit"] != 2
                or len(result["gens"]) != 2 or result.get("unfinished_chains", 0)
                or not result.get("same_vocab", True)):
            raise ValueError(f"Incomplete or incompatible smoke artifact: {path}")
        for row in result["gens"]:
            if (len(row["texts"]) != 2 or any(not text.strip() for text in row["texts"])
                    or len(row["correct"]) != 2
                    or any(grade not in (0, 1) for grade in row["correct"])):
                raise ValueError(f"Invalid smoke generations: {path}")


def save_status(path: Path, record: dict) -> None:
    """Atomically publish monitoring state; full results remain provisional until passed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
