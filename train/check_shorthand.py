"""Fail closed on unusable calibration or incomplete solo-pilot training."""
import argparse
import hashlib
import json
from pathlib import Path

from figure2_checkpoint import finite, read_training_records, verify_weights

TASKS = ("manipulate_matrix", "string_manipulation", "ruletaker_shared", "rearc_objects")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibration", type=Path)
    mode.add_argument("--training", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS[:2]))
    parser.add_argument("--budgets", type=int, nargs=2, default=[256, 1024])
    args = parser.parse_args()
    if args.calibration:
        if args.data_root is None:
            parser.error("--data-root required for calibration")
        result = check_calibration(args.calibration, args.data_root, args.tasks, args.budgets)
        path = args.calibration / "verified.json"
    else:
        if args.task is None:
            parser.error("--task required for training")
        result = check_training(args.training, args.steps, args.task)
        path = args.training / "verified.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def check_calibration(root: Path, data_root: Path, tasks=TASKS[:2], budgets=(256, 1024)) -> dict:
    """Require calibrated difficulty, usable format, and full training-data integrity.

    The longer-budget base accuracy must lie strictly between 0 and 0.98, with
    at least 50% complete answer blocks. The shorter-budget arm must have at least one success, so training has
    a nonzero correctness signal. All-zero rewards require a revised budget.
    Check all prompt lengths using the same base tokenizer as evaluation.
    """
    if len(budgets) != 2 or not 0 < budgets[0] < budgets[1]:
        raise ValueError("Require two increasing positive token budgets")
    from transformers import AutoTokenizer
    import pandas as pd
    summary = {}
    tokenizer = None
    for task in tasks:
        results = [json.loads((root / f"{task}-calibration-b{budget}.json").read_text())
                   for budget in budgets]
        if any(result["provenance"]["max_tokens"] != budget for result, budget in zip(results, budgets)):
            raise ValueError("Calibration token budget mismatch")
        score = results[1]["metrics"]
        if (not 0 < score["accuracy"] < 0.98 or score["answer_present"] < 0.5
                or results[0]["metrics"]["accuracy"] == 0):
            raise ValueError(f"Task needs recalibration before training: {task}: {score}")
        if task in {"ruletaker_shared", "rearc_objects"}:
            check_concept_signal(task, score)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(results[1]["provenance"]["model"])
        manifest = json.loads((data_root / task / "manifest.json").read_text())
        calibration_hash = manifest["splits"]["calibration"]["sha256"]
        if any(r["provenance"]["data_sha256"] != calibration_hash for r in results):
            raise ValueError("Calibration was evaluated on different data")
        max_prompt = 0
        for split, spec in manifest["splits"].items():
            path = data_root / task / f"{split}.parquet"
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError(f"Data hash mismatch: {path}")
            rows = pd.read_parquet(path)
            if len(rows) != spec["rows"]:
                raise ValueError(f"Data length mismatch: {path}")
            # Test integrity only: no answer scores or test-driven tuning.
            for prompt in rows["prompt"]:
                ids = tokenizer.apply_chat_template(list(prompt), tokenize=True, add_generation_prompt=True)
                max_prompt = max(max_prompt, len(ids))
        prompt_limit = min(1536, 4096 - budgets[1])
        if max_prompt > prompt_limit:
            raise ValueError(f"Prompt exceeds training budget: {task}: {max_prompt}")
        summary[task] = {"calibration": [r["metrics"] for r in results], "max_prompt_tokens": max_prompt}
    return {"verified": True, "tasks": summary, "budgets": list(budgets)}


def check_concept_signal(task: str, metrics: dict) -> None:
    """Reject truncation-confounded or unlearnable new pilots before allocating training."""
    if metrics["truncated"] >= 0.10 or metrics["answer_present"] < 0.90:
        raise ValueError(f"Generous-budget calibration still truncates: {task}: {metrics}")
    if task == "ruletaker_shared" and metrics["query_accuracy"] <= 0.55:
        raise ValueError("RuleTaker calibration must exceed the 50% random-query baseline")
    if task == "rearc_objects" and any(v["accuracy"] == 0 for v in metrics["by_family"].values()):
        raise ValueError("At least one Re-ARC family has no calibration successes; revise using calibration only")


def check_training(root: Path, steps: int, task: str) -> dict:
    """Require all updates, finite gradients, final validation and intact final weights."""
    records = {step: metrics for step, metrics in read_training_records(root).items()
               if 1 <= step <= steps}
    if set(records) != set(range(1, steps + 1)):
        raise ValueError("Incomplete training step history")
    for step, metrics in records.items():
        finite(metrics.get("actor/pg_loss"), f"policy loss at {step}")
        finite(metrics.get("actor/grad_norm"), f"gradient at {step}")
        if "actor/tandem_senior_token_frac" in metrics:
            raise ValueError("Solo pilot unexpectedly used tandem rollouts")
    metric = f"val-core/{task}/acc/best@4/mean"
    score = finite(records[steps].get(metric), "final validation")
    if not 0 <= score <= 1:
        raise ValueError("Invalid validation probability")
    checkpoint = root / "hf" / f"global_step_{steps}"
    verify_weights(checkpoint)
    return {"verified": True, "path": str(checkpoint), "steps": steps,
            "selection": "fixed final checkpoint", "validation_metric": metric,
            "validation_pass4": score}


if __name__ == "__main__":
    main()
