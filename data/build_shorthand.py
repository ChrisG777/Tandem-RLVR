"""Build immutable, disjoint repeated-operation tasks from pinned Reasoning Gym."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pandas as pd
import reasoning_gym

REVISION = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
TASKS = ("manipulate_matrix", "string_manipulation")
SPLITS = {"calibration": 64, "train": 4096, "validation": 128, "test": 256, "long": 256}
INSTRUCTION = (
    "\nSolve the problem step by step. End with the final answer between <answer> and </answer>. "
    "Inside those tags, write only the requested matrix (one row per line, entries separated "
    "by spaces) or string, without commentary or a code block."
)
MATRIX_CONVENTIONS = (
    "All rotations are clockwise. A horizontal mirror reverses row order; a vertical mirror "
    "reverses column order. A diagonal mirror transposes the matrix. A counterdiagonal mirror "
    "reverses both row and column order and then transposes.\n\n"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.task, args.out_dir, args.seed), indent=2))


def build_dataset(task: str, out_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Write all splits and provenance; refuse overwrites or generator-revision drift.

    Reuse upstream oracles. Split seed/index ranges do not overlap; prompt hashes
    are deduplicated globally. Metadata retains the oracle's intermediate states.
    """
    distribution = importlib.metadata.distribution("reasoning-gym")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != REVISION:
        raise ValueError("Use the pinned env/shorthand-data uv project")
    if task not in TASKS:
        raise ValueError(task)
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"task": task, "revision": REVISION, "seed": seed,
                                "source": "https://github.com/open-thought/reasoning-gym", "splits": {}}
    seen: set[str] = set()
    for split_no, (split, count) in enumerate(SPLITS.items()):
        config = generator_config(task, split == "long")
        source_seed = seed + 1_000_000 * (split_no + 1)
        dataset = reasoning_gym.create_dataset(task, seed=source_seed, size=100_000, **config)
        rows = []
        rejected = 0
        for index in range(100_000):
            entry = dataset[index]
            metadata = entry["metadata"]
            if not eligible(task, entry, split == "long"):
                rejected += 1
                continue
            content = (MATRIX_CONVENTIONS if task == "manipulate_matrix" else "") + entry["question"] + INSTRUCTION
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in seen:
                rejected += 1
                continue
            seen.add(digest)
            rows.append({
                "data_source": task,
                "prompt": [{"role": "user", "content": content}],
                "ability": "algorithmic",
                "reward_model": {"ground_truth": entry["answer"], "style": "rule"},
                "extra_info": {"index": len(rows), "split": split, "source_seed": source_seed,
                               "source_index": index, "prompt_sha256": digest,
                               "oracle_json": json.dumps(metadata, sort_keys=True)},
            })
            if len(rows) == count:
                break
        if len(rows) != count:
            raise RuntimeError(f"Insufficient eligible examples for {task}/{split}")
        path = out_dir / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        manifest["splits"][split] = {"rows": count, "seed": source_seed, "config": config,
                                     "examined": index + 1, "rejected": rejected,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "max_prompt_chars": max(len(r["prompt"][0]["content"]) for r in rows)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def generator_config(task: str, longer: bool) -> dict[str, Any]:
    """Return upstream generator kwargs; negative weights disable via softmax underflow."""
    if task == "manipulate_matrix":
        return dict(min_rows=3, max_rows=4, min_cols=3, max_cols=4,
                    min_transforms=12 if longer else 6, max_transforms=16 if longer else 10,
                    w_crop=-1000, w_remove_every_nth_row=-1000,
                    w_remove_every_nth_col=-1000, w_zero_divisible=-1000)
    return dict(min_string_length=12, max_string_length=24, min_num_rules=8, max_num_rules=12)


def eligible(task: str, entry: dict[str, Any], longer: bool) -> bool:
    """Exclude trivial/unchanged answers, 360-degree rotations, and out-of-range string chains."""
    meta = entry["metadata"]
    if task == "manipulate_matrix":
        return (meta["matrix"] != meta["solution"]
                and all(op.get("degrees") != "360" for op in meta["operations"]))
    count = len(meta["states"]) - 1
    lower, upper = (12, 18) if longer else (4, 10)
    return lower <= count <= upper and bool(entry["answer"])


if __name__ == "__main__":
    main()
