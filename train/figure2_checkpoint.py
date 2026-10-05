"""Verify independent training artifacts and select by held-out pass@4 only."""
import argparse
import json
import math
from pathlib import Path
import struct

VALIDATION = "val-core/deepscaler/acc/best@4/mean"


def main() -> None:
    """Validate a completed run and atomically save its selected checkpoint record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("grpo", "tandem"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
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
    records = {}
    for path in sorted(root.glob("metrics-*.jsonl"), key=lambda p: p.stat().st_mtime_ns):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            step, data = row["step"], row["data"]
            if 1 <= step <= steps:
                records.setdefault(step, {}).update(data)
    if set(records) != set(range(1, steps + 1)):
        raise ValueError(f"Missing training steps in {root}")
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
            "training_steps": steps, "verified": True}


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
