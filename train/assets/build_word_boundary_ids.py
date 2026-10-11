#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUT = Path(__file__).resolve().parent / "qwen3_word_boundary_ids.json"

LEADING_SPACE_MARKER = "Ġ"

EXPECTED_COUNT = 53021


def build(model: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return sorted(
        token_id
        for surface, token_id in tokenizer.get_vocab().items()
        if surface.startswith(LEADING_SPACE_MARKER)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-count", type=int, help="Optional pinned tokenizer boundary count")
    parser.add_argument("--write", action="store_true", help="overwrite --out instead of checking it")
    args = parser.parse_args()

    ids = build(args.model)
    expected = args.expected_count if args.expected_count is not None else (EXPECTED_COUNT if args.model == DEFAULT_MODEL else None)
    if not ids or (expected is not None and len(ids) != expected):
        raise SystemExit(f"Unexpected boundary count for {args.model}: expected={expected}, actual={len(ids)}")

    payload = json.dumps(ids)

    if args.write:
        args.out.write_text(payload)
        print(f"wrote {len(ids)} ids to {args.out}")
        return

    if not args.out.exists():
        raise SystemExit(f"{args.out} does not exist; rerun with --write")
    if args.out.read_text() != payload:
        raise SystemExit(f"{args.out} differs from what {args.model} produces")
    print(f"{args.out} matches {args.model}: {len(ids)} ids")


if __name__ == "__main__":
    main()
