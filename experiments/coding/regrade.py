"""Regrade saved APPS responses in isolation without rerunning inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import common
from experiments.coding.campaign import preflight
from experiments.coding.reward import compute_score, grader_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    preflight()
    regrade(args.input, args.out, data=args.data, workers=args.workers)


def regrade(source: Path, out: Path, *, data: Path, workers: int = 4) -> dict:
    """Rescore saved validation responses; resume per problem, preserve original grades.

    Inputs, dataset, and grader identities must match on resume. Infrastructure
    failures raise, leaving the last atomic progress file. Never overwrite source.
    """
    if source.resolve() == out.resolve() or workers < 1:
        raise ValueError("Use a separate output and at least one worker")
    raw = source.read_bytes()
    result = json.loads(raw)
    dataset_hash = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    if result["dataset_sha256"] != dataset_hash:
        raise ValueError("Dataset differs from saved responses")
    previous_data = common.DATA
    try:
        common.DATA = data / "validation"
        problems = common.load_problems(tuple(result["benchmarks"]))
    finally:
        common.DATA = previous_data
    rows = result["gens"]
    if len(rows) != len(problems) or any(
        (r["set"], r["task_id"]) != (p["set"], p["task_id"])
        or len(r["texts"]) != result["n"] or len(r["correct"]) != result["n"]
        for r, p in zip(rows, problems)
    ):
        raise ValueError("Incomplete or mismatched saved responses")
    identity = {"source_sha256": hashlib.sha256(raw).hexdigest(),
                "dataset_sha256": dataset_hash, "grader": grader_identity()}
    progress_path, progress = common.load_progress(str(out), identity, problems, batch_size=1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, problem in zip(rows[len(progress["gens"]):], problems[len(progress["gens"]):]):
            diagnostics = list(pool.map(
                lambda text: compute_score("apps", text, problem["gt"]), row["texts"]))
            progress["gens"].append({**row, "original_correct": row["correct"],
                "original_diagnostics": row["diagnostics"], "diagnostics": diagnostics,
                "correct": [d["acc"] for d in diagnostics]})
            common.save_json(progress_path, progress)
            print(f"Regraded {len(progress['gens'])}/{len(rows)} problems", flush=True)
    result["original_grader"] = result["grader"]
    result["original_metrics"] = result.get("metrics")
    result["grader"] = identity["grader"]
    result["gens"] = progress["gens"]
    result["metrics"] = common.metrics_by_set(
        problems, [int(sum(r["correct"])) for r in result["gens"]], result["n"])
    result["regrading"] = {**identity, "source": str(source.resolve()),
        "changed_rewards": sum(a != b for r in result["gens"]
                               for a, b in zip(r["original_correct"], r["correct"]))}
    common.save_json(out, result)
    return result


if __name__ == "__main__":
    main()
