"""Prepare revision-pinned APPS splits for the existing verl trainer."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

BASE = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TIERS = ("introductory", "interview")
INSTRUCTION = "Solve the following programming problem step by step. Put your final complete Python solution inside a single ```python code block."


def build_apps(out_dir: Path, *, revision: str, seed: int = 0) -> dict:
    """Write train/validation and tiered eval Parquets plus hashes; never overwrite.

    Requires a full immutable Hub commit. Test rows exceeding 4096 prompt tokens
    remain in evaluation; training/validation rows are filtered before splitting.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pandas as pd
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision must be an immutable 40-character commit")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(out_dir)
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    # The pinned source has JSONL plus a legacy dataset script; use the maintained
    # JSON loader instead of executing that script (unsupported by datasets 4).
    files = {split: hf_hub_download("codeparrot/apps", f"{split}.jsonl",
                                  repo_type="dataset", revision=revision)
             for split in ("train", "test")}
    ds = load_dataset("json", data_files=files)
    splits, excluded = prepare_splits(ds, tok, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in splits.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "task_ids": [r["extra_info"]["task_id"] for r in rows],
                       "rows": len(rows)}
    manifest = {"dataset": "codeparrot/apps", "revision": revision, "seed": seed,
                "tokenizer": BASE, "tokenizer_revision": BASE_REVISION,
                "max_prompt": 4096, "max_response": 3000, "context": 8192,
                "files": files, "excluded": excluded}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_splits(dataset, tokenizer, *, seed: int = 0, validation_per_tier: int = 100):
    """Convert and deduplicate before seeded splitting; return rows and audit."""
    from experiments.coding.reward import normalize_tests
    excluded, converted = [], {"train": [], "test": []}
    test_statements = {" ".join(r["question"].split()) for r in dataset["test"]}
    seen = set()
    for split in ("test", "train"):
        for r in sorted(dataset[split], key=lambda r: r["problem_id"]):
            if r["difficulty"] not in TIERS:
                continue
            task_id = f"{split}/{r['problem_id']}"
            statement = " ".join(r["question"].split())
            reason = None
            if split == "train" and statement in test_statements:
                reason = "test_duplicate"
            elif split == "train" and statement in seen:
                reason = "train_duplicate"
            try:
                tests = normalize_tests(r["input_output"])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                reason = reason or f"malformed_tests: {exc}"
            if reason:
                excluded.append({"task_id": task_id, "reason": reason})
                continue
            if split == "train":
                seen.add(statement)
            starter = r.get("starter_code") or ""
            mode = ("Implement the supplied function signature."
                    if tests.get("fn_name") else "Read from standard input and write to standard output.")
            content = f"{r['question']}\n\n{mode}"
            if starter:
                content += f"\n\nStarter code:\n```python\n{starter}\n```"
            content += "\n\n" + INSTRUCTION
            prompt = [{"role": "user", "content": content}]
            tokens = tokenizer.apply_chat_template(prompt, tokenize=True,
                                                   add_generation_prompt=True, enable_thinking=False)
            if split == "train" and len(tokens) > 4096:
                excluded.append({"task_id": task_id, "reason": "overlength", "tokens": len(tokens)})
                continue
            converted[split].append({"data_source": "apps", "prompt": prompt,
                "ability": "coding", "reward_model": {"style": "rule", "ground_truth": json.dumps(tests)},
                "extra_info": {"task_id": task_id, "index": r["problem_id"],
                    "difficulty": r["difficulty"], "statement": r["question"], "starter_code": starter,
                    "prompt_tokens": len(tokens), "overlength": len(tokens) > 4096}})
    rng, train, validation = random.Random(seed), [], []
    for tier in TIERS:
        rows = [r for r in converted["train"] if r["extra_info"]["difficulty"] == tier]
        if len(rows) <= validation_per_tier:
            raise ValueError(f"Insufficient {tier} training rows")
        held = set(rng.sample(range(len(rows)), validation_per_tier))
        train.extend(r for i, r in enumerate(rows) if i not in held)
        validation.extend(r for i, r in enumerate(rows) if i in held)
    result = {"train.parquet": train, "validation.parquet": validation}
    for split, rows in (("test", converted["test"]), ("validation", validation)):
        for tier in TIERS:
            result[f"{split}/apps_{tier}/test.parquet"] = [r for r in rows if r["extra_info"]["difficulty"] == tier]
    return result, excluded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build_apps(args.out_dir, revision=args.revision, seed=args.seed), indent=2))
