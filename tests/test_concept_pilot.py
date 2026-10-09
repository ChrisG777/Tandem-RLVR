"""Check actual oracle labels, strict partial credit, split isolation and launch gates."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "reward"), str(ROOT / "train")]
from shorthand_reward import compute_score
from check_shorthand import check_concept_signal


class ConceptPilot(unittest.TestCase):
    def test_fractional_reward_keeps_exact_accuracy_separate(self):
        result = compute_score("ruletaker_shared", "<answer>true false false true</answer>",
                               "true true false true")
        self.assertEqual(result["score"], .75)
        self.assertEqual(result["acc"], 0)
        for answer in ("true false", "true false false true true", "True false false true"):
            self.assertEqual(compute_score("ruletaker_shared", f"<answer>{answer}</answer>",
                                          "true true false true")["score"], 0)
        self.assertEqual(compute_score("rearc_objects", "<answer>1 2\n3 4</answer>", "1 2 3 4")["score"], 0)

    def test_gate_rejects_truncation_chance_and_dead_families(self):
        metrics = {"truncated": .05, "answer_present": .95, "query_accuracy": .8,
                   "by_family": {"a": {"accuracy": .3}, "b": {"accuracy": .2}}}
        for task in ("ruletaker_shared", "rearc_objects"):
            check_concept_signal(task, metrics)
            with self.assertRaises(ValueError):
                check_concept_signal(task, {**metrics, "truncated": .10})
        with self.assertRaises(ValueError):
            check_concept_signal("ruletaker_shared", {**metrics, "query_accuracy": .5})
        with self.assertRaises(ValueError):
            check_concept_signal("rearc_objects", {**metrics, "by_family": {"a": {"accuracy": 0}}})

    def test_saved_data_and_independent_oracles(self):
        import pandas as pd
        from problog import get_evaluatable
        from problog.program import PrologString
        from reasoning_gym.arc.rearc_utils import verifiers
        for task in ("ruletaker_shared", "rearc_objects"):
            root = ROOT / "data/concept-pilot" / task
            manifest = json.loads((root / "manifest.json").read_text())
            seen_prompts, seen_instances = set(), set()
            for split, spec in manifest["splits"].items():
                path = root / f"{split}.parquet"
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), spec["sha256"])
                rows = pd.read_parquet(path).to_dict("records")
                self.assertEqual(len(rows), spec["rows"])
                for row in rows:
                    meta = row["extra_info"]
                    oracle = json.loads(meta["oracle_json"])
                    digest = hashlib.sha256(row["prompt"][0]["content"].encode()).hexdigest()
                    self.assertEqual(digest, meta["prompt_sha256"])
                    self.assertNotIn(digest, seen_prompts)
                    seen_prompts.add(digest)
                    for key in oracle["instance_keys"]:
                        self.assertNotIn(key, seen_instances)
                        seen_instances.add(key)
                    self.assertLessEqual(meta["prompt_tokens"], 1024)
                    target = row["reward_model"]["ground_truth"]
                    self.assertEqual(compute_score(task, f"<answer>{target}</answer>", target)["score"], 1)
                    if task == "rearc_objects":
                        verify = getattr(verifiers, f"verify_{oracle['family']}")
                        for example in oracle["examples"]:
                            self.assertEqual(verify(tuple(map(tuple, example["input"]))),
                                             tuple(map(tuple, example["output"])))
                    # Re-evaluate a deterministic sample per split using ProbLog.
                    elif meta["index"] < 4:
                        values = {str(k): bool(v) for k, v in get_evaluatable().create_from(
                            PrologString(oracle["program"])).evaluate().items()}
                        answers = ["true" if values[f"{q['predicate']}({','.join(q['arguments'])})"] else "false"
                                   for q in oracle["queries"]]
                        self.assertEqual(" ".join(answers), target)


if __name__ == "__main__":
    unittest.main()
