"""Reward and data-integrity checks for the solo shorthand pilot."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reward"))
from shorthand_reward import compute_score


class ShorthandReward(unittest.TestCase):
    def test_preserves_matrix_shape(self):
        good = "Reasoning. <answer>1   2\n3 4</answer>"
        self.assertEqual(compute_score("manipulate_matrix", good, "1 2\n3 4")["score"], 1)
        self.assertEqual(compute_score("manipulate_matrix", good, "1 2 3 4")["score"], 0)

    def test_no_substring_or_multiple_answer_credit(self):
        for text in ("abc", "<answer>xabc</answer>", "<answer>abc</answer> trailing",
                     "<answer>abc</answer><answer>abc</answer>", "<answer>abc"):
            self.assertEqual(compute_score("string_manipulation", text, "abc")["score"], 0)
        self.assertEqual(compute_score("string_manipulation", "<answer>abc</answer>", "abc")["score"], 1)

    def test_unknown_task_fails(self):
        with self.assertRaises(ValueError):
            compute_score("unknown", "<answer>a</answer>", "a")


class DatasetIntegrity(unittest.TestCase):
    def test_saved_splits(self):
        import pandas as pd
        seen = set()
        for task in ("manipulate_matrix", "string_manipulation"):
            root = ROOT / "data/shorthand" / task
            manifest = json.loads((root / "manifest.json").read_text())
            for split, spec in manifest["splits"].items():
                path = root / f"{split}.parquet"
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), spec["sha256"])
                rows = pd.read_parquet(path)
                self.assertEqual(len(rows), spec["rows"])
                for row in rows.to_dict("records"):
                    digest = hashlib.sha256(row["prompt"][0]["content"].encode()).hexdigest()
                    self.assertEqual(digest, row["extra_info"]["prompt_sha256"])
                    self.assertNotIn(digest, seen)
                    seen.add(digest)
                    answer = row["reward_model"]["ground_truth"]
                    self.assertEqual(compute_score(task, f"<answer>{answer}</answer>", answer)["score"], 1)
                    oracle = json.loads(row["extra_info"]["oracle_json"])
                    if task == "string_manipulation":
                        lo, hi = (12, 18) if split == "long" else (4, 10)
                        self.assertTrue(lo <= len(oracle["states"]) - 1 <= hi)
                    else:
                        self.assertNotEqual(oracle["matrix"], oracle["solution"])
                        self.assertFalse(any(op.get("degrees") == "360" for op in oracle["operations"]))


if __name__ == "__main__":
    unittest.main()
