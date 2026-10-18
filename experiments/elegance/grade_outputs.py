"""Score saved solo/handoff evaluation outputs with the training elegance reward."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pandas as pd

from experiments.elegance.reward import compute_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="eval/solo.py or eval/handoff.py JSON")
    parser.add_argument("--data", type=Path, required=True, help="Elegance split parquet the result was run on")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    summary = grade_outputs(args.result, args.data, args.out, workers=args.workers)
    print(json.dumps(summary, indent=2))


def grade_outputs(result: Path, data: Path, out: Path, *, workers: int = 4) -> dict:
    """Grade every saved sample with `compute_score`; write per-sample grades and rates.

    Inputs: an evaluation JSON whose `gens[i].texts` follow the parquet's row
    order, and that parquet (question/references in `extra_info`). Requires
    ELEGANCE_JUDGE_CACHE and ELEGANCE_CODEX_BIN; correct answers each make one
    cached Codex call. Judge failures raise, so no sample is silently ungraded.
    Output: {"samples": [...], "summary": {...}} with rates over all samples,
    and the elegant share of correct samples. Atomic write; nothing else changes.
    """
    rows = pd.read_parquet(data)
    gens = json.loads(result.read_text())["gens"]
    if len(gens) != len(rows):
        raise ValueError(f"{len(gens)} generations for {len(rows)} problems")
    jobs = [(int(g["idx"]), k, text) for g in gens for k, text in enumerate(g["texts"])]

    def grade(job):
        idx, k, text = job
        row = rows.iloc[idx]
        extra = dict(row["extra_info"])
        score = compute_score(row["data_source"], text, row["reward_model"]["ground_truth"], extra)
        return {"idx": idx, "sample": k, "problem_id": extra.get("problem_id"),
                **{key: float(score[key]) for key in ("acc", "valid_derivation", "elegant", "score")}}

    with ThreadPoolExecutor(workers) as pool:
        samples = list(pool.map(grade, jobs))
    summary = summarize(samples)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps({"result": str(result), "data": str(data), "samples": samples, "summary": summary}, indent=2))
    tmp.replace(out)
    return summary


def summarize(samples: list[dict]) -> dict:
    """Sample-weighted rates plus problem-macro elegant rate."""
    n = len(samples)
    correct = sum(s["acc"] for s in samples)
    elegant = sum(s["elegant"] for s in samples)
    by_problem: dict[int, list[float]] = {}
    for s in samples:
        by_problem.setdefault(s["idx"], []).append(s["elegant"])
    return {
        "samples": n,
        "accuracy": correct / n,
        "elegant_rate": elegant / n,
        "valid_derivation_rate": sum(s["valid_derivation"] for s in samples) / n,
        "elegant_of_correct": elegant / correct if correct else None,
        "problem_macro_elegant_rate": sum(sum(v) / len(v) for v in by_problem.values()) / len(by_problem),
    }


if __name__ == "__main__":
    main()
