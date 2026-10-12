"""Download MedCalc-Bench-Verified and write the verl parquet splits.

The official test split is evaluation only. Checkpoint selection uses a
calculator-stratified holdout drawn from the official train split.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

import common
from experiments.medcalc.protocol import (
    BASE_MODEL,
    BASE_REVISION,
    CATEGORIES,
    CATEGORY_TITLES,
    DATASET_ID,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    HOLDOUT_N,
    MAX_PROMPT,
    SPLIT_SEED,
    holdout_mask,
    slugify,
)
from experiments.medcalc.rows import make_record


def download_base(out: Path) -> Path:
    path = Path(snapshot_download(
        BASE_MODEL,
        revision=BASE_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.jinja", "*.model"],
    ))
    record = {"repo_id": BASE_MODEL, "revision": BASE_REVISION, "path": str(path)}
    _write_json(out / "base.json", record)
    print(f"base weights: {path}")
    return path


def prompt_lengths(records: list[dict], tokenizer_path: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    lengths = []
    for record in records:
        prompt = common.chat_prefix(tokenizer, record["prompt"][0]["content"])
        lengths.append(len(tokenizer.encode(prompt)))
    over = sum(length > MAX_PROMPT for length in lengths)
    summary = {
        "tokenizer": tokenizer_path,
        "max_prompt": MAX_PROMPT,
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "over_max_prompt": over,
    }
    print(
        f"prompt tokens: min {summary['min']}, mean {summary['mean']:.0f}, "
        f"max {summary['max']}, over {MAX_PROMPT}: {over}/{len(lengths)}"
    )
    if over:
        print(
            f"warning: verl will drop {over} training/held-out prompts longer than "
            f"{MAX_PROMPT} tokens. Raise MAX_PROMPT in experiments/medcalc/protocol.py "
            "and the train/eval launchers together if that share is large."
        )
    return summary


def write_parquet(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel_for(grouped: dict[str, list[dict]]) -> dict:
    order = [slugify(name) for name in CATEGORIES if slugify(name) in grouped]
    order += sorted(set(grouped) - set(order))
    benchmarks = {}
    for name in order:
        task_ids = [record["extra_info"]["task_id"] for record in grouped[name]]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"Duplicate test task ids in {name}")
        benchmarks[name] = {
            "title": CATEGORY_TITLES.get(name, name),
            "task_ids": task_ids,
        }
    return {"benchmarks": benchmarks}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "medcalc")
    parser.add_argument("--revision", default=None, help="immutable dataset Hub commit")
    parser.add_argument("--holdout-n", type=int, default=HOLDOUT_N)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--tokenizer", default=None, help="count prompt tokens with this tokenizer")
    parser.add_argument("--download-base", action="store_true")
    parser.add_argument("--allow-changed-counts", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.download_base:
        downloaded = download_base(args.out)
        if args.tokenizer is None:
            args.tokenizer = str(downloaded)

    revision = args.revision or HfApi().dataset_info(DATASET_ID).sha
    dataset = load_dataset(DATASET_ID, revision=revision)
    counts = {name: len(dataset[name]) for name in ("train", "test")}
    print("rows:", counts, "revision:", revision)
    expected = {"train": EXPECTED_TRAIN, "test": EXPECTED_TEST}
    if counts != expected and not args.allow_changed_counts:
        raise SystemExit(
            f"row counts {counts} != {expected}. The Hub dataset moved. "
            "Pass --allow-changed-counts only after checking the release notes."
        )

    train_rows = [make_record(dict(row), index) for index, row in enumerate(dataset["train"])]
    test_rows = [make_record(dict(row), index) for index, row in enumerate(dataset["test"])]
    mask = holdout_mask(
        [row["extra_info"]["calculator_id"] for row in train_rows],
        args.holdout_n,
        args.seed,
    )
    train = [row for row, held in zip(train_rows, mask) if not held]
    heldout = [row for row, held in zip(train_rows, mask) if held]
    train_ids = {row["extra_info"]["index"] for row in train}
    held_ids = {row["extra_info"]["index"] for row in heldout}
    if train_ids & held_ids or len(train_ids) + len(held_ids) != len(train_rows):
        raise SystemExit("train/held-out split does not partition the official train rows")

    hashes = {
        "train.parquet": write_parquet(args.out / "train.parquet", train),
        "heldout.parquet": write_parquet(args.out / "heldout.parquet", heldout),
    }
    grouped: dict[str, list[dict]] = {}
    for row in test_rows:
        grouped.setdefault(row["extra_info"]["category_slug"], []).append(row)
    panel = panel_for(grouped)
    for name, rows in grouped.items():
        relative = f"eval/{name}/test.parquet"
        hashes[relative] = write_parquet(args.out / relative, rows)

    manifest = {
        "dataset": DATASET_ID,
        "revision": revision,
        "split_seed": args.seed,
        "holdout_rows": len(heldout),
        "train_rows": len(train),
        "test_rows": len(test_rows),
        "official_counts": counts,
        "eval_benchmarks": list(panel["benchmarks"]),
        "scoring": (
            "Boxed final answer only. Decimal answers must fall in the row's "
            "[lower, upper] band; integers, dates, and weeks/days follow "
            "MedCalc-Bench-Verified evaluation/evaluate.py. Model text is not eval'd."
        ),
        "sha256": hashes,
    }
    if args.tokenizer:
        manifest["prompt_tokens"] = {
            "train": prompt_lengths(train, args.tokenizer),
            "heldout": prompt_lengths(heldout, args.tokenizer),
            "test": prompt_lengths(test_rows, args.tokenizer),
        }
    _write_json(args.out / "manifest.json", manifest)
    _write_json(args.out / "panel.json", panel)
    print(f"wrote {len(train)} train + {len(heldout)} held out + {len(test_rows)} test -> {args.out}")
    print("sample gold:", train[0]["reward_model"]["ground_truth"])


if __name__ == "__main__":
    main()
