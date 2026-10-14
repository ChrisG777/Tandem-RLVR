"""One-time, audited resume migration for the APPS output-limit worker fix."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import common
import config
from experiments.coding.reward import grader_identity


def main() -> None:
    """Preserve completed grades; only the formerly fatal output exception changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("grpo", "tandem"), required=True)
    parser.add_argument("--phase", choices=("solo", "handoff", "word-handoff"), required=True)
    parser.add_argument("--concurrency", type=int, default=8, help="Qualified native-word execution concurrency")
    args = parser.parse_args()
    root = args.root.resolve()
    old_runtime, new_runtime = root / "grader", root / "grader-output-fix"
    repo = Path(__file__).resolve().parents[2]
    original = subprocess.check_output(["git", "show", "f0b035766bfd55ee773cf7ae3514ab61b343ee18:experiments/coding/worker.py"], cwd=repo).decode()
    expected = original.replace(
        "except (SystemExit, KeyboardInterrupt, MemoryError):",
        "except (SystemExit, KeyboardInterrupt, MemoryError, OutputLimitExceeded):").replace(
        "class BoundedIO(io.StringIO):",
        'class OutputLimitExceeded(RuntimeError):\n    """Candidate output exhausted the sandbox\'s existing output allowance."""\n\n\nclass BoundedIO(io.StringIO):').replace(
        'raise RuntimeError("Output limit exceeded")',
        'raise OutputLimitExceeded("Output limit exceeded")')
    assert (old_runtime / "worker.py").read_text() == original
    assert (new_runtime / "worker.py").read_text() == expected
    os.environ["APPS_GRADER_RUNTIME"] = str(old_runtime)
    old_grader = grader_identity()
    os.environ["APPS_GRADER_RUNTIME"] = str(new_runtime)
    new_grader = grader_identity()
    assert {k: v for k, v in old_grader["sha256"].items() if k != "worker.py"} == {
        k: v for k, v in new_grader["sha256"].items() if k != "worker.py"}
    report = json.loads((root / "calibration/report.json").read_text())
    assert report["passed"] and report["grader"] == old_grader
    common.DATA = root / "data/test"
    common.MAX_MODEL_LEN, common.MAX_PROMPT, config.MAX_TOKENS = 12288, 4096, 6000
    problems = common.load_problems(("apps_introductory", "apps_interview"))
    model = json.loads((root / f"{args.arm}-full/selected.json").read_text())["path"]
    identity = {**common.evaluation_identity(problems), "phase": args.phase,
                "n": 32 if args.phase == "solo" else 8, "sampling": config.record()}
    if args.phase == "solo":
        identity.update(model=model, seed=17)
    elif args.phase == 'handoff':
        from transformers import AutoTokenizer
        junior = json.loads((root / "base.json").read_text())["base"]["path"]
        same_vocab = AutoTokenizer.from_pretrained(model).get_vocab() == AutoTokenizer.from_pretrained(junior).get_vocab()
        identity.update(senior=model, junior=junior, max_rounds=150, same_vocab=same_vocab)
    else:
        assert args.concurrency > 0
        junior = json.loads((root / "base.json").read_text())["base"]["path"]
        boundary = repo / 'train/assets/qwen3_word_boundary_ids.json'
        identity.update(senior=model, junior=junior, decoding='evaluation',
            schedule={'selection_strategy': 'word', 'prob_primary': 0.5,
                      'max_gap_tokens': 32, 'boundary_sha256': hashlib.sha256(boundary.read_bytes()).hexdigest()},
            seed_rule='1000 * problem_index + sample_index', max_model_len=12288,
            max_num_seqs=args.concurrency,
            kv_cache_memory_bytes_per_model=5*1024**3*math.ceil(args.concurrency*12288/16384),
            budget_rule='min(6000, 12288 - prompt_tokens - 8)')
    batch_size = 1 if args.phase == 'word-handoff' else common.EVAL_BATCH_SIZE
    out = root / "results" / args.arm / f"{args.phase}.json"
    assert not out.exists(), "Do not migrate completed evaluations"
    path = Path(str(out) + ".progress.json")
    backup = Path(str(path) + ".before-output-fix")
    receipt = Path(str(path) + ".output-fix-migration.json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved["new_grader"] == new_grader
        common.load_progress(str(out), identity, problems, batch_size=batch_size)
        print("Already migrated", path)
        return
    raw = path.read_bytes()
    old_identity = {**identity, "grader": old_grader}
    if args.phase == 'word-handoff':
        old_identity['max_num_seqs'] = 2
        del old_identity['kv_cache_memory_bytes_per_model']
    _, progress = common.load_progress(str(out), old_identity, problems, batch_size=batch_size)
    old_signature = progress["signature"]
    if backup.exists():
        assert backup.read_bytes() == raw
    else:
        with backup.open("xb") as f:
            f.write(raw)
    progress["signature"] = hashlib.sha256(json.dumps(
        {"identity": identity, "problems": problems, "batch_size": batch_size}, sort_keys=True).encode()).hexdigest()
    provenance = {"old_grader": old_grader, "new_grader": new_grader,
        "old_signature": old_signature, "new_signature": progress["signature"],
        "source_sha256": hashlib.sha256(raw).hexdigest(), "preserved_problems": len(progress["gens"]),
        "old_identity": old_identity, "new_identity": identity,
        "execution_change": "Native concurrency/cache may change batching and sampled trajectories; model, seeds, sampling distribution, budgets and word-switch rules are unchanged. Saved responses are retained, not regenerated.",
        "compatibility": "Only OutputLimitExceeded changes from fatal infrastructure error to candidate runtime_error; previously returned grades are unchanged."}
    progress["grader_migration"] = provenance
    common.save_json(path, progress)
    common.load_progress(str(out), identity, problems, batch_size=batch_size)
    common.save_json(receipt, provenance)
    print(json.dumps({"path": str(path), "preserved_problems": len(progress["gens"])}))


if __name__ == "__main__":
    main()
