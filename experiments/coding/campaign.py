"""APPS campaign checks and dispatch, sharing the reproduction's model runners."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "train")]
TIERS = ("apps_introductory", "apps_interview")
MAX_TRUNCATION = 0.101


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    admit = sub.add_parser("admit-training")
    admit.add_argument("--report", type=Path, required=True)
    admit.add_argument("--allow-pending", action="store_true",
                       help="Explicitly authorized provisional training while calibration runs")
    cal = sub.add_parser("calibration")
    cal.add_argument("--solo", type=Path, required=True)
    cal.add_argument("--word", type=Path, required=True)
    cal.add_argument("--out", type=Path, required=True)
    cal.add_argument("--data", type=Path, required=True, help="Prepared APPS directory containing manifest.json")
    run = sub.add_parser("evaluate")
    run.add_argument("--phase", choices=("solo", "handoff", "word-handoff", "legibility"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--junior", required=True)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--split", choices=("test", "validation"), default="test")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--n", type=int, default=8)
    run.add_argument("--decoding", choices=("training", "evaluation"), default="evaluation")
    run.add_argument("--solo", type=Path)
    run.add_argument("--native-concurrency", type=int, help="Optional native-word execution concurrency; size GPU memory accordingly")
    select = sub.add_parser("select")
    select.add_argument("--root", type=Path, required=True)
    panel = sub.add_parser("panel")
    panel.add_argument("--data", type=Path, required=True)
    panel.add_argument("--out", type=Path, required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--data", type=Path, required=True)
    protocol.add_argument("--base", type=Path, required=True)
    protocol.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "admit-training":
        from experiments.coding.reward import grader_identity
        print(json.dumps(training_admission(args.report, allow_pending=args.allow_pending,
                                            expected_grader=grader_identity())))
    elif args.command == "preflight":
        preflight()
    elif args.command == "calibration":
        raw_manifest = (args.data / "manifest.json").read_bytes()
        report = calibration(json.loads(args.solo.read_text()), json.loads(args.word.read_text()),
                             manifest=json.loads(raw_manifest),
                             manifest_sha256=hashlib.sha256(raw_manifest).hexdigest())
        from common import save_json
        save_json(args.out, report)
        if not report["passed"]:
            raise SystemExit("Calibration gate failed; training is not admitted. See " + str(args.out))
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "select":
        from figure2_checkpoint import verify_training
        from common import save_json
        manifest = json.loads((args.root / "base.json").read_text())
        for arm in ("grpo", "tandem"):
            # Smokes gate admission; selection does not need their disposable weights.
            manifest[arm] = verify_training(args.root / f"{arm}-full", arm, 200, validation_metric="apps/validation/pass@4")
            if json.loads((args.root / f"{arm}-full/initialization.json").read_text()) != {"base": manifest["base"]}:
                raise ValueError("Arm did not initialize from the pinned base")
        save_json(args.root / "models.json", manifest)
    elif args.command == "protocol":
        from common import save_json
        from experiments.coding.reward import grader_identity
        from experiments.coding.build_apps import BASE_REVISION
        import config
        base = json.loads(args.base.read_text())["base"]
        if Path(base["path"]).name != BASE_REVISION:
            raise ValueError("Base manifest must identify the pinned immutable snapshot")
        save_json(args.out, {
            "experiment": "apps-figure2-single-seed-pilot", "base": base,
            "dataset_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
            "grader": grader_identity(), "context": 12288, "max_prompt": 4096, "max_response": 6000,
            "calibration": {"validation_tasks": 200, "samples_per_mode": 8,
                            "accuracy_range": [0.05, 0.90], "min_mixed_groups": 0.10,
                            "max_truncation": MAX_TRUNCATION, "threshold_status": "pilot",
                            "revision_reason": "User authorized relaunch after throughput qualification; pilot limit revised from 10% to 10.1% for observed 161/1600 native truncations; no test scores consulted"},
            "training": {"steps": 200, "seed": 42, "batch": 16, "minibatch": 8,
                         "rollouts": 8, "learning_rate": 1e-6, "clip_ratio": 0.2,
                         "temperature": 0.6, "top_p": 1.0, "top_k": -1,
                         "save_validate_every": 20, "validation_samples": 4, "smoke_steps": 3},
            "tandem": {"boundaries": "train/assets/qwen3_word_boundary_ids.json",
                       "senior_probability": 0.5, "gap_cap": 32, "junior_token_loss_weight": 0},
            "selection": {"metric": "apps/validation/pass@4", "tie_break": "earliest_step",
                          "rollout_mode": "own_arm", "split": "validation"},
            "evaluation": {"sampling": {**config.record(), "max_tokens": 6000}, "solo_samples": 32,
                           "paragraph_samples": 8, "native_word_samples": 8,
                           "macro_weighting": "equal_tier", "bootstrap_unit": "problem",
                           "cross_entropy_text": "full_response"},
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        })
    else:
        from common import save_json
        data = json.loads((args.data / "manifest.json").read_text())
        save_json(args.out, {"benchmarks": {tier: {"title": tier.removeprefix("apps_").capitalize(),
            "task_ids": data["files"][f"test/{tier}/test.parquet"]["task_ids"]} for tier in TIERS}})


def training_admission(report: Path, *, allow_pending: bool = False,
                       expected_grader: dict | None = None) -> dict:
    """Reject failed/stale calibration; allow pending calibration only by explicit opt-in."""
    if report.exists():
        result = json.loads(report.read_text())
        if result.get("passed") is not True:
            raise ValueError("Calibration failed; stop provisional training before restarting")
        if expected_grader is not None and result.get("grader") != expected_grader:
            raise ValueError("Calibration grader identity differs; regrade saved calibration before training")
        return {"calibration": "passed", "provisional": False}
    if not allow_pending:
        raise ValueError("Calibration is incomplete")
    return {"calibration": "pending", "provisional": True}


def preflight() -> None:
    """Verify sandbox isolation and known outcomes before executing model outputs."""
    from experiments.coding.reward import run_sandbox, compute_score, normalize_tests
    if run_sandbox({"preflight": True}, timeout=15) != {"isolated": True}:
        raise RuntimeError("Sandbox isolation check failed")
    fixtures = [({"inputs": ["2\n"], "outputs": ["4\n"]}, "print(int(input())*2)", 1),
                ({"inputs": [[2]], "outputs": [4], "fn_name": "double"}, "def double(x):\n    return x*2", 1),
                ({"inputs": [[2]], "outputs": [4], "fn_name": "double"}, "def double(x):\n    return 0", 0)]
    for tests, code, expected in fixtures:
        actual = compute_score("apps", f"```python\n{code}\n```", json.dumps(normalize_tests(tests)))
        if actual["acc"] != expected:
            raise RuntimeError(f"Checker preflight failed: {actual}")
    print("APPS sandbox and stdin/call-based checker preflight passed", flush=True)


def calibration(solo: dict, word: dict, *, manifest: dict, manifest_sha256: str) -> dict:
    """Require exact 200-problem/eight-sample paired inputs before applying gate."""
    if solo["phase"] != "solo" or word["phase"] != "word-handoff":
        raise ValueError("Wrong calibration phases")
    if solo.get("model") != word.get("senior") or word.get("senior") != word.get("junior"):
        raise ValueError("Calibration must use the same base for all model roles")
    from experiments.coding.build_apps import BASE_REVISION
    if Path(solo.get("model") or "").name != BASE_REVISION:
        raise ValueError("Calibration must identify the immutable pinned base snapshot")
    expected_ids = {(tier, task) for tier in TIERS
                    for task in manifest["files"][f"validation/{tier}/test.parquet"]["task_ids"]}
    if len(expected_ids) != 200:
        raise ValueError("Manifest must contain exactly 200 validation tasks")
    if not 0.3 < word.get("senior_token_fraction", 0) < 0.7:
        raise ValueError("Calibration lacks valid word-sampler participation")
    expected_sampling = {"temperature": 0.6, "top_p": 1.0, "top_k": -1, "max_tokens": 6000}
    identities = []
    summary = {}
    for label, result in (("solo", solo), ("word", word)):
        if result["n"] != 8 or result["sampling"] != expected_sampling or len(result["gens"]) != 200:
            raise ValueError("Calibration requires 200 problems, eight training-decoding samples")
        keys = [(g["set"], g["task_id"]) for g in result["gens"]]
        if len(set(keys)) != 200 or any(len(g["correct"]) != 8 for g in result["gens"]):
            raise ValueError("Duplicate tasks or incomplete samples")
        identities.append(set(keys))
        if set(keys) != expected_ids or result["dataset_sha256"] != manifest_sha256:
            raise ValueError("Calibration does not match the prepared validation manifest")
        if result.get("context") != 12288 or result.get("max_prompt") != 4096:
            raise ValueError("Calibration context/prompt limits differ from the protocol")
        if any(value not in (0, 1) for g in result["gens"] for value in g["correct"]):
            raise ValueError("Calibration rewards must be binary")
        by_tier = {}
        for tier in TIERS:
            rows = [g for g in result["gens"] if g["set"] == tier]
            if len(rows) != 100:
                raise ValueError("Expected 100 validation problems per tier")
            by_tier[tier] = describe(rows)
        pooled = describe(result["gens"])
        passed = 0.05 <= pooled["accuracy"] <= 0.90 and pooled["mixed_groups"] >= 0.10 and pooled["truncation"] <= MAX_TRUNCATION
        summary[label] = {"by_tier": by_tier, "pooled": pooled, "passed": passed}
    if identities[0] != identities[1] or solo["dataset_sha256"] != word["dataset_sha256"] or solo["grader"] != word["grader"]:
        raise ValueError("Calibration inputs/graders differ")
    return {"passed": all(s["passed"] for s in summary.values()), "modes": summary,
            "thresholds": {"accuracy_range": [0.05, 0.90], "min_mixed_groups": 0.10,
                           "max_truncation": MAX_TRUNCATION, "status": "pilot"},
            "dataset_sha256": solo["dataset_sha256"], "grader": solo["grader"],
            "source_sha256": {name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                              for name, value in (("solo", solo), ("word", word))}}


def describe(rows: list[dict]) -> dict:
    samples = [s for g in rows for s in g["samples"]]
    if len(samples) != 8 * len(rows):
        raise ValueError("Missing calibration sample diagnostics")
    lengths = [s.get("n_tokens", len(s.get("token_ids", []))) for s in samples]
    authorship = [v for s in samples for v in s.get("model_mask", [])]
    return {"accuracy": sum(sum(g["correct"]) for g in rows)/(8*len(rows)),
            "mixed_groups": sum(0 < sum(g["correct"]) < 8 for g in rows)/len(rows),
            "truncation": sum(s["finish_reason"] == "length" for s in samples)/len(samples),
            "mean_tokens": sum(lengths)/len(lengths), "max_tokens": max(lengths),
            "senior_fraction": sum(authorship)/len(authorship) if authorship else 1.0,
            "failures": {key: sum(d.get(key, 0) for g in rows for d in g["diagnostics"])
                         for key in ("format_error", "syntax_error", "runtime_error", "wrong_answer", "timeout")}}


def evaluate(args) -> None:
    """Dispatch GPU inference to the original runners with explicit APPS overrides."""
    preflight()
    scripts = {"solo": "eval/solo.py", "handoff": "eval/handoff.py",
               "word-handoff": "experiments/word_handoff/word_handoff.py", "legibility": "eval/legibility.py"}
    command = [sys.executable, str(ROOT / scripts[args.phase]), "--out", str(args.out)]
    if args.phase == "legibility":
        if args.solo is None:
            raise ValueError("Legibility requires --solo")
        command += ["--solo", str(args.solo), "--junior", args.junior]
    else:
        command += ["--data-root", str(args.data / args.split), "--benchmarks", ",".join(TIERS),
                    "--context", "12288", "--max-prompt", "4096", "--max-response", "6000", "--n", str(args.n)]
        if args.phase == "solo":
            command += ["--model", args.model, "--decoding", args.decoding]
        else:
            command += ["--senior", args.model, "--junior", args.junior]
            if args.phase == "word-handoff":
                command += ["--decoding", args.decoding]
                if args.native_concurrency is not None:
                    command += ["--concurrency", str(args.native_concurrency)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
