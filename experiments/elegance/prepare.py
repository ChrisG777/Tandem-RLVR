"""Build a fixed joint method holdout from the published audit archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import zipfile

TARGETS = ("complement", "vieta", "power_point")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.out), indent=2))


def prepare(archive: Path, destination: Path) -> dict:
    """Write verl Parquets, references and immutable split/hash manifest.

    Joint training pool is the INTERSECTION of the three safe training pools;
    test is their union, never used for training-time validation. Twenty seeded
    in-family items form validation. References appear only in reward metadata,
    never in policy prompts. Refuse nonempty output directories.
    """
    import pandas as pd

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
        for name in ("dataset.json", "holdout-folds.json"):
            if hashlib.sha256(z.read(name)).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Corrupt release: {name}")
        dataset = json.loads(z.read("dataset.json"))
        folds = json.loads(z.read("holdout-folds.json"))["folds"]
    rows = {
        row["problem"]["id"]: row
        for row in dataset["problems"]
        if row["training_eligible"]
    }
    chosen = [next(f for f in folds if f["target"] == target) for target in TARGETS]
    safe = set.intersection(*(set(f["train_ids"]) for f in chosen))
    test = set.union(*(set(f["test_ids"]) for f in chosen))
    if safe & test or len(safe) < 36:
        raise ValueError("Invalid or undersized split")
    validation = set(random.Random(42).sample(sorted(safe), 20))
    splits = {
        "train": sorted(safe - validation),
        "validation": sorted(validation),
        "test": sorted(test),
    }
    excluded_methods = sorted(
        set.union(*(set(f["excluded_training_methods"]) for f in chosen))
    )
    for pid in safe:
        if any(
            set(route["methods"]) & set(excluded_methods)
            for route in rows[pid]["problem"]["routes"]
        ):
            raise ValueError("Held-out method leaked into train/validation")
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, ids in splits.items():
        records = [training_record(rows[pid]) for pid in ids]
        path = destination / f"{split}.parquet"
        pd.DataFrame(records).to_parquet(path, index=False)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "seed": 42,
        "targets": list(TARGETS),
        "excluded_training_methods": excluded_methods,
        "splits": splits,
        "counts": {k: len(v) for k, v in splits.items()},
        "files": files,
        "excluded_problem_ids": sorted(set(rows) - safe - test),
        "reward": "correct + 0.5 * correct * valid_derivation * elegant",
        "judge_model": "gpt-6.1-sol",
        "test_during_training": False,
    }
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def training_record(row: dict) -> dict:
    """Convert one eligible audit record to verl's existing dataset contract."""
    p = row["problem"]
    refs = [
        {
            "text": r["text"],
            "elegant": r["elegance"]["elegant"],
            "reason": r["elegance"]["reason"],
        }
        for r in p["routes"]
        if r["correctness"] == "valid"
    ]
    return {
        "data_source": "method_elegance",
        "ability": "math",
        "prompt": [
            {
                "role": "user",
                "content": p["question"]
                + "\n\nSolve the problem step by step. Put your final answer in \\boxed{}.",
            }
        ],
        "reward_model": {"style": "rule", "ground_truth": str(p["answer"])},
        "extra_info": {
            "problem_id": p["id"],
            "question": p["question"],
            "references": json.dumps(refs, ensure_ascii=False),
        },
    }


if __name__ == "__main__":
    main()
