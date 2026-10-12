"""Exact APPS checkpoint pass@4 from complete, manifest-matched validation traces."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from common import pass_at_k


def validation_pass4(root: Path, *, steps: int) -> dict[int, float]:
    """Validate every expected checkpoint's four-response groups and return pass@4.

    Read only validation data and saved traces. Full runs require all 200 tasks;
    three-step smokes permit eight manifest tasks or all 200. Missing/foreign tasks,
    partial groups, changed data, and nonbinary rewards raise. No logs are rewritten.
    """
    data = root.parent / "data"
    raw = (root / "data-manifest.json").read_bytes()
    if raw != (data / "manifest.json").read_bytes():
        raise ValueError("Validation manifest differs from training")
    info = json.loads(raw)["files"]["validation.parquet"]
    parquet = data / "validation.parquet"
    if hashlib.sha256(parquet.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("Validation data hash differs")
    problems = pd.read_parquet(parquet).to_dict("records")
    ids = [p["extra_info"]["task_id"] for p in problems]
    if len(ids) != 200 or len(set(ids)) != 200 or ids != info["task_ids"]:
        raise ValueError("Expected the exact 200 validation tasks")
    expected_steps = {3} if steps == 3 else {*range(20, steps + 1, 20), steps}
    scores = {}
    for step in sorted(expected_steps):
        path = root / "validation" / f"{step}.jsonl"
        groups = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["step"] != step or row["score"] not in (0, 1) or row["acc"] != row["score"]:
                raise ValueError("Invalid validation step or reward")
            groups.setdefault(row["input"], []).append(row)
        allowed = (8, 200) if steps == 3 else (200,)
        if len(groups) not in allowed or any(len(g) != 4 for g in groups.values()):
            raise ValueError("Incomplete validation tasks or four-sample groups")
        observed = {}
        for prompt, rows in groups.items():
            matches = [p for p in problems if p["prompt"][0]["content"] in prompt
                       and all(r["gts"] == p["reward_model"]["ground_truth"] for r in rows)]
            if len(matches) != 1:
                raise ValueError("Validation trace does not match a unique validation task")
            task = matches[0]["extra_info"]["task_id"]
            if task in observed:
                raise ValueError("Duplicate validation task")
            observed[task] = pass_at_k(4, int(sum(r["score"] for r in rows)), 4)
        if not set(observed) <= set(ids) or (len(groups) == 200 and set(observed) != set(ids)):
            raise ValueError("Validation task membership differs")
        scores[step] = sum(observed.values()) / len(observed)
    return scores
