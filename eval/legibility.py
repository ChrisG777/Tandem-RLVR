import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import common
import config

DEFAULT_JUNIOR = "Qwen/Qwen3-4B-Instruct-2507"


def aggregate_ce(per_sample):
    by_set = {}
    for row in per_sample:
        by_set.setdefault(row["set"], []).append(row["ce"])
    by_benchmark = common.group_by_benchmark(by_set)
    ces = [row["ce"] for row in per_sample]
    return {
        "by_set": {name: float(np.mean(v)) for name, v in by_set.items()},
        "by_benchmark": {name: float(np.mean(v)) for name, v in by_benchmark.items()},
        "mean_ce": float(np.mean(ces)) if ces else None,
        "macro_ce": (
            float(np.mean([np.mean(v) for v in by_benchmark.values()]))
            if by_benchmark else None
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", required=True, help="the JSON solo.py wrote")
    ap.add_argument("--junior", default=DEFAULT_JUNIOR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=0, help="CoTs per problem, 0 = all")
    ap.add_argument("--gpu-util", type=float, default=0.60)
    ap.add_argument("--max-batched-tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=2000, help="prompts per generate call")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    solo = common.load_json(args.solo)
    common.DATA = common.Path(solo.get("data_root", common.DATA))
    common.MAX_MODEL_LEN = solo.get("context", common.MAX_MODEL_LEN)
    problems = common.problems_from_result(solo)
    by_idx = {g["idx"]: g for g in solo["gens"]}

    tok = AutoTokenizer.from_pretrained(args.junior)

    todo, dropped = [], 0
    for p in problems:
        prefix = common.chat_prefix(tok, p["content"])
        prefix_len = len(tok(prefix).input_ids)
        texts = by_idx[p["idx"]]["texts"]
        if args.samples:
            texts = texts[: args.samples]
        for sample_idx, cot in enumerate(texts):
            full = prefix + cot
            if len(tok(full).input_ids) >= common.MAX_MODEL_LEN - common.CONTEXT_RESERVE:
                dropped += 1
                continue
            todo.append((p, sample_idx, full, prefix_len))
    print(f"[plan] scoring {len(todo)} CoTs, dropped {dropped} that exceed the context")

    llm = common.build_engine(
        args.junior, args.gpu_util, max_num_batched_tokens=args.max_batched_tokens
    )
    params = config.scoring_params()

    progress_path = Path(str(args.out) + ".progress.json")
    signature = hashlib.sha256(json.dumps({"solo": solo, "junior": args.junior,
        "samples": args.samples, "context": common.MAX_MODEL_LEN}, sort_keys=True).encode()).hexdigest()
    progress = {"signature": signature, "completed": 0, "per_sample": []}
    if progress_path.exists():
        progress = common.load_json(progress_path)
        if progress["signature"] != signature:
            raise ValueError("Cross-entropy inputs differ from saved progress")
    per_sample = progress["per_sample"]
    for start in range(progress["completed"], len(todo), args.chunk):
        batch = todo[start : start + args.chunk]
        outs = llm.generate([t[2] for t in batch], params)
        for (p, sample_idx, _, prefix_len), o in zip(batch, outs):
            nll = [
                -list(d.values())[0].logprob
                for d in o.prompt_logprobs[prefix_len:]
                if d
            ]
            if nll:
                per_sample.append(
                    {
                        "set": p["set"],
                        "idx": p["idx"],
                        "sample_idx": sample_idx,
                        "ce": float(np.mean(nll)),
                        "n_tokens": len(nll),
                    }
                )
        progress["completed"] = min(start + args.chunk, len(todo))
        common.save_json(progress_path, progress)
        print(f"[progress] {progress['completed']}/{len(todo)} responses scored", flush=True)

    result = {
        "phase": "legibility",
        "scored_text": "full_response",
        "source_sha256": signature,
        "senior": solo["model"],
        "junior": args.junior,
        "solo_src": args.solo,
        "benchmarks": solo["benchmarks"],
        "limit": solo.get("limit", 0),
        "n_scored": len(per_sample),
        "n_dropped": dropped,
        **aggregate_ce(per_sample),
        "per_sample": per_sample,
    }
    common.save_json(args.out, result)
    print(f"LEGIBILITY {solo['model']} read by {args.junior}")
    if result["macro_ce"] is None:
        raise SystemExit("no CoT was scored; every one exceeded the context window")
    print(f"  macro mean CE = {result['macro_ce']:.4f} nats over {len(per_sample)} CoTs")


if __name__ == "__main__":
    main()
