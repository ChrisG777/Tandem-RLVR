"""Verify independent training artifacts and select by held-out pass@4 only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import re

VALIDATION = "val-core/deepscaler/acc/best@4/mean"


def main() -> None:
    """Validate a completed run and atomically save its selected checkpoint record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("grpo", "tandem"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--mode", choices=("smoke", "full"))
    parser.add_argument("--record-resume-step", type=int)
    parser.add_argument("--resume-log", type=Path)
    args = parser.parse_args()
    if args.record_resume_step is not None:
        if args.resume_log is None:
            parser.error("--resume-log is required to record evidence")
        print(json.dumps(record_resume_gap(args.root, args.record_resume_step, args.resume_log), indent=2))
        return
    if args.arm is None or args.steps is None or args.mode is None:
        parser.error("verification requires --arm, --steps and --mode")
    selected = verify_training(args.root, args.arm, args.steps)
    selected["mode"] = args.mode
    temporary = args.root / "selected.partial"
    temporary.write_text(json.dumps(selected, indent=2) + "\n")
    temporary.replace(args.root / "selected.json")


def verify_training(root: Path, arm: str, steps: int) -> dict:
    """Require finite losses, a live Tandem mask, validation and complete HF weights.

    Read per-attempt verl JSONL files, allowing checkpoint resumes. Return the best
    held-out checkpoint (earliest step breaks ties); raise on incomplete artifacts.
    Test-benchmark scores never participate in checkpoint selection.
    """
    records = {step: data for step, data in read_training_records(root).items() if step <= steps}
    gaps = sorted(set(range(1, steps + 1)) - records.keys())
    for step in gaps:
        if step in (1, steps) or step - 1 not in records or step + 1 not in records:
            raise ValueError(f"Unexplained training gap at step {step} in {root}")
        path = root / "resume-evidence" / f"step-{step}.json"
        if not path.exists():
            raise ValueError(f"Missing training step {step}; no resume evidence in {root}")
        _verify_resume_evidence(root, step, json.loads(path.read_text()))
    for step, data in records.items():
        finite(data.get("actor/pg_loss"), f"policy loss at {step}")
        if "actor/grad_norm" in data:
            finite(data["actor/grad_norm"], f"gradient norm at {step}")
        if arm == "tandem":
            fraction = finite(data.get("actor/tandem_senior_token_frac"), f"Tandem mask at {step}")
            if not 0.3 < fraction < 0.7:
                raise ValueError(f"Tandem senior fraction is not near 0.5 at {step}: {fraction}")
    candidates = []
    for step, data in records.items():
        if VALIDATION in data:
            score = finite(data[VALIDATION], f"validation at {step}")
            if not 0 <= score <= 1:
                raise ValueError(f"Invalid validation probability {score}")
            checkpoint = root / "hf" / f"global_step_{step}"
            verify_weights(checkpoint)
            candidates.append((score, -step, checkpoint))
    if not candidates or VALIDATION not in records[steps]:
        raise ValueError("Final held-out validation is missing")
    score, negative_step, checkpoint = max(candidates)
    return {"arm": arm, "path": str(checkpoint.resolve()), "step": -negative_step,
            "validation_metric": VALIDATION, "validation_pass4": score,
            "training_steps": steps, "verified": True,
            "unobserved_metric_steps": gaps,
            "verification_scope": "Observed metrics and validated checkpoint/resume evidence; missing metrics are not imputed"}


def read_training_records(root: Path) -> dict[int, dict]:
    """Read complete JSONL attempts, discarding abandoned steps when a retry rewinds.

    Later attempts replace whole step records, so stale validation values from an
    abandoned attempt cannot attach to a new checkpoint at the same step.
    Malformed logs fail rather than silently drop data.
    """
    return merge_training_records([rows for _, rows in _read_attempts(root)])


def merge_training_records(attempts: list[list[dict]]) -> dict[int, dict]:
    """Merge chronological attempt rows, replacing records from a resumed step onward."""
    records = {}
    for rows in attempts:
        rows = [row for row in rows if row["step"] >= 1]
        if not rows:
            continue
        first = rows[0]["step"]
        records = {step: data for step, data in records.items() if step < first}
        for row in rows:
            records[row["step"]] = row["data"]
    return records


def record_resume_gap(root: Path, step: int, log: Path) -> dict:
    """Save narrowly scoped evidence of a checkpoint saved before its metrics logged.

    Require adjacent attempt boundaries, intact saved HF weights, and explicit
    model/optimizer/RNG/scheduler load messages for this run's one-rank checkpoint.
    Save only evidence lines and the source log hash; never create metrics.
    """
    attempts = _read_attempts(root)
    boundary = next(((before.name, after.name)
                     for (before, old), (after, new) in zip(attempts, attempts[1:])
                     if old and new and old[-1]["step"] == step - 1 and new[0]["step"] == step + 1), None)
    if boundary is None:
        raise ValueError("No adjacent pre/post-resume metric boundary for this gap")
    raw = log.read_bytes()
    clean = re.sub(r"\x1b\[[0-9;]*m", "", raw.decode(errors="replace"))
    expected = _resume_messages(root, step)
    lines = [next((line for line in clean.splitlines() if message in line), "") for message in expected]
    evidence = {"step": step, "root": str(root.resolve()), "before": boundary[0], "after": boundary[1],
                "source_log": str(log.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
                "load_messages": lines,
                "limitation": "Metrics and validation for this step were not observed and remain unavailable"}
    _verify_resume_evidence(root, step, evidence)
    directory = root / "resume-evidence"
    directory.mkdir(exist_ok=True)
    path = directory / f"step-{step}.json"
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(evidence, indent=2) + "\n")
    temporary.replace(path)
    return evidence


def _verify_resume_evidence(root: Path, step: int, evidence: dict) -> None:
    if evidence.get("root") != str(root.resolve()) or evidence.get("step") != step:
        raise ValueError("Resume evidence belongs to another run/step")
    attempts = _read_attempts(root)
    boundaries = [(before.name, after.name) for (before, old), (after, new) in zip(attempts, attempts[1:])
                  if old and new and old[-1]["step"] == step - 1 and new[0]["step"] == step + 1]
    if (evidence.get("before"), evidence.get("after")) not in boundaries:
        raise ValueError("Resume evidence does not match metric attempt boundaries")
    lines = evidence.get("load_messages", [])
    if len(lines) != 4 or any(message not in line for message, line in zip(_resume_messages(root, step), lines)):
        raise ValueError("Incomplete checkpoint-load evidence")
    verify_weights(root / "hf" / f"global_step_{step}")


def _resume_messages(root: Path, step: int) -> list[str]:
    actor = root.resolve() / f"global_step_{step}" / "actor"
    return [f"[Rank 0] Loaded {component} from {actor}/{filename}_world_size_1_rank_0.pt"
            for component, filename in (("model", "model"), ("optimizer", "optim"),
                                        ("rng", "extra_state"), ("lr_scheduler", "extra_state"))]


def _read_attempts(root: Path) -> list[tuple[Path, list[dict]]]:
    return [(path, [json.loads(line) for line in path.read_text().splitlines() if line.strip()])
            for path in sorted(root.glob("metrics-*.jsonl"), key=lambda p: p.stat().st_mtime_ns)]


def finite(value: object, label: str) -> float:
    """Return a finite numeric metric, rejecting JSON null and NaN."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite {label}: {value}")
    return float(value)


def verify_weights(path: Path) -> None:
    """Validate config and safetensors shard lengths without loading GPU tensors."""
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError(f"Unexpected model config: {path}")
    index = path / "model.safetensors.index.json"
    shards = ([path / name for name in set(json.loads(index.read_text())["weight_map"].values())]
              if index.exists() else list(path.glob("*.safetensors")))
    if not shards:
        raise ValueError(f"No saved weights: {path}")
    for shard in shards:
        with shard.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            if header_bytes > 100_000_000:
                raise ValueError(f"Invalid safetensors header: {shard}")
            header = json.loads(handle.read(header_bytes))
        tensors = [v for k, v in header.items() if k != "__metadata__"]
        if not tensors or 8 + header_bytes + max(v["data_offsets"][1] for v in tensors) != shard.stat().st_size:
            raise ValueError(f"Truncated checkpoint shard: {shard}")


if __name__ == "__main__":
    main()
