"""Prepare matched, blinded trace pairs; never infer jargon from length alone."""
import argparse
import json
from pathlib import Path
from random import Random



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_review(args.run_root), indent=2))


def build_review(run_root: Path, seed: int = 61) -> dict:
    """Write metrics, blinded pairs, and a separate answer key from completed evaluations.

    Compare identical prompts at identical evaluation budgets. Uniform pairs use
    sample zero without correctness filtering; the explicitly labeled successful
    subset uses each model's first correct sample. No model judge is called.
    """
    rng = Random(seed)
    output = run_root / "review"
    output.mkdir(exist_ok=True)
    pairs, key, summary = [], {}, []
    for directory in sorted((run_root / "eval").iterdir()):
        if not directory.is_dir() or directory.name == "base":
            continue
        for path in sorted(directory.glob("*.json")):
            base = json.loads((run_root / "eval/base" / path.name).read_text())
            trained = json.loads(path.read_text())
            for field in ("data_sha256", "max_tokens", "seed", "n", "temperature", "top_p"):
                if base["provenance"][field] != trained["provenance"][field]:
                    raise ValueError(f"Unmatched evaluation field {field}: {path}")
            base_by_id = {g["prompt_sha256"]: g for g in base["generations"]}
            train_by_id = {g["prompt_sha256"]: g for g in trained["generations"]}
            if base_by_id.keys() != train_by_id.keys():
                raise ValueError("Evaluation problem mismatch")
            summary.append({"training": directory.name, "evaluation": path.stem,
                            "base": base["metrics"], "trained": trained["metrics"],
                            "accuracy_change": trained["metrics"]["accuracy"] - base["metrics"]["accuracy"]})
            all_ids = sorted(base_by_id)
            both_correct = [i for i in all_ids if any(s["acc"] for s in base_by_id[i]["samples"])
                            and any(s["acc"] for s in train_by_id[i]["samples"])]
            for subset, candidates in (("uniform", all_ids), ("both_correct", both_correct)):
                for identity in rng.sample(candidates, min(8, len(candidates))):
                    base_row, train_row = base_by_id[identity], train_by_id[identity]
                    samples = []
                    for label, row in (("base", base_row), ("trained", train_row)):
                        sample = (row["samples"][0] if subset == "uniform" else
                                  next(s for s in row["samples"] if s["acc"]))
                        samples.append((label, sample))
                    rng.shuffle(samples)
                    pair_id = f"pair-{len(pairs):04d}"
                    key[pair_id] = {"training": directory.name, "evaluation": path.stem,
                                    "A": samples[0][0], "B": samples[1][0]}
                    pairs.append({"id": pair_id, "subset": subset, "prompt": base_row["prompt"],
                                  "target": base_row["target"], "A": samples[0][1], "B": samples[1][1]})
    if len(summary) != 16:
        raise ValueError(f"Expected all 16 trained evaluations, found {len(summary)}")
    report = {"seed": seed, "comparisons": summary, "pairs": len(pairs),
              "interpretation": "Descriptive pilot. Blinded semantic review remains required; no jargon conclusion is automated."}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "blinded_pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    (output / "answer_key.json").write_text(json.dumps(key, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
