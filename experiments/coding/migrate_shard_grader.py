"""Audited re-keying of APPS evaluation shards to the alarm-timeout worker fix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import common
import config
from experiments.coding.reward import grader_identity

COMPATIBILITY = ("Only an escaped per-test alarm changes: formerly a fatal infrastructure error, now "
                 "a candidate timeout, and the worker disarms the checker's leftover alarm. Saved "
                 "rewards are unchanged; a rare zero-reward diagnostic could differ.")


def main() -> None:
    """Re-key complete or partial shards from --old-runtime to --new-runtime grading.

    Each shard's stored base must equal the reconstructed old-grader base (else raise);
    already migrated shards are skipped. Originals are backed up beside each shard.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("grpo", "tandem"), required=True)
    parser.add_argument("--phase", choices=("solo", "handoff", "word-handoff"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--junior", required=True)
    parser.add_argument("--old-runtime", type=Path, required=True)
    parser.add_argument("--new-runtime", type=Path, required=True)
    args = parser.parse_args()
    graders = {}
    for name, runtime in (("old", args.old_runtime), ("new", args.new_runtime)):
        os.environ["APPS_GRADER_RUNTIME"] = str(runtime.resolve())
        graders[name] = grader_identity()
    unchanged = {k: v for k, v in graders["old"]["sha256"].items() if k != "worker.py"}
    assert unchanged == {k: v for k, v in graders["new"]["sha256"].items() if k != "worker.py"}
    common.DATA = (args.root / "data/test").resolve()
    common.MAX_MODEL_LEN, common.MAX_PROMPT, config.MAX_TOKENS = 12288, 4096, 6000
    problems = common.load_problems(("apps_introductory", "apps_interview"))
    bases = {name: common.shard_base(identity(args, problems, grader), problems)
             for name, grader in graders.items()}
    out = args.root / "results" / args.arm / f"{args.phase}.json"
    for path in sorted(out.parent.glob(out.name + ".shard-*-*.progress.json")):
        raw = path.read_bytes()
        shard = json.loads(raw)
        if shard["base"] == bases["new"]:
            continue
        if shard["base"] != bases["old"]:
            raise ValueError(f"Shard is not from the expected old grader: {path}")
        backup = Path(str(path) + ".before-timeout-fix")
        with backup.open("xb") as f:
            f.write(raw)
        shard["base"] = bases["new"]
        shard["grader_migration"] = {"old_grader": graders["old"], "new_grader": graders["new"],
                                     "old_base": bases["old"], "new_base": bases["new"],
                                     "source_sha256": hashlib.sha256(raw).hexdigest(),
                                     "preserved_problems": len(shard["gens"]),
                                     "compatibility": COMPATIBILITY}
        common.save_json(path, shard)
        print(json.dumps({"migrated": str(path), "preserved_problems": len(shard["gens"])}), flush=True)


def identity(args, problems: list[dict], grader: dict) -> dict:
    """Reconstruct each evaluator's protocol identity (execution keys are excluded later)."""
    base ={**common.evaluation_identity(problems), "grader": grader}
    sampling = config.record()
    if args.phase == "solo":
        return {**base, "phase": "solo", "model": args.model, "n": 32, "sampling": sampling, "seed": 17}
    if args.phase == "handoff":
        from transformers import AutoTokenizer
        same = AutoTokenizer.from_pretrained(args.model).get_vocab() == AutoTokenizer.from_pretrained(args.junior).get_vocab()
        return {**base, "phase": "handoff", "senior": args.model, "junior": args.junior, "n": 8,
                "sampling": sampling, "max_rounds": 150, "same_vocab": same}
    boundary = Path(__file__).resolve().parents[2] / "train/assets/qwen3_word_boundary_ids.json"
    return {**base, "phase": "word-handoff", "senior": args.model, "junior": args.junior, "n": 8,
            "sampling": sampling, "decoding": "evaluation",
            "schedule": {"selection_strategy": "word", "prob_primary": 0.5, "max_gap_tokens": 32,
                         "boundary_sha256": hashlib.sha256(boundary.read_bytes()).hexdigest()},
            "seed_rule": "1000 * problem_index + sample_index", "max_model_len": 12288,
            "budget_rule": "min(6000, 12288 - prompt_tokens - 8)"}


if __name__ == "__main__":
    main()
