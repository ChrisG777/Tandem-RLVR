import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.elegance import judge, reward


class EleganceRewardTests(unittest.TestCase):
    def test_published_joint_split_and_prompt_isolation(self):
        import json
        import pandas as pd
        from experiments.elegance.prepare import prepare

        archive = (
            Path(__file__).resolve().parents[1] / "data/method-elegance/dataset.zip"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "prepared"
            manifest = prepare(archive, root)
            self.assertEqual(
                manifest["counts"], {"train": 152, "validation": 20, "test": 17}
            )
            train, validation, test = (
                set(manifest["splits"][k]) for k in ("train", "validation", "test")
            )
            self.assertFalse(train & validation or train & test or validation & test)
            record = pd.read_parquet(root / "train.parquet").iloc[0]
            self.assertNotIn("references", record["prompt"][0]["content"])
            self.assertTrue(json.loads(record["extra_info"]["references"]))

    def test_reward_is_answer_and_proof_gated(self):
        context = {"question": "Q", "references": []}
        with (
            patch.dict("os.environ", {"ELEGANCE_JUDGE_CACHE": "/tmp/test-unused"}),
            patch.object(
                reward,
                "answer_score",
                return_value={"score": 1.0, "acc": 1.0, "answer_present": 1.0},
            ),
            patch.object(
                reward,
                "judge_solution",
                return_value={"valid": True, "elegant": True, "reason": "shortcut"},
            ) as model,
        ):
            self.assertEqual(
                reward.compute_score("method_elegance", "proof", "1", context)["score"],
                1.5,
            )
            model.return_value = {
                "valid": False,
                "elegant": False,
                "reason": "answer only",
            }
            self.assertEqual(
                reward.compute_score("method_elegance", "1", "1", context)["score"], 1.0
            )
            model.side_effect = RuntimeError("quota")
            with self.assertRaisesRegex(RuntimeError, "quota"):
                reward.compute_score("method_elegance", "proof", "1", context)
        with (
            patch.object(
                reward, "answer_score", return_value={"score": 0.0, "acc": 0.0}
            ),
            patch.object(reward, "judge_solution") as model,
        ):
            self.assertEqual(
                reward.compute_score("method_elegance", "wrong", "1")["score"], 0.0
            )
            model.assert_not_called()

    def test_cache_identity_and_failure_semantics(self):
        request = {"question": "Q", "candidate": "C", "answer": "A", "references": []}
        grade = {"valid": True, "elegant": False, "reason": "enumeration"}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(judge, "invoke_codex", return_value=(grade, "test")) as call,
        ):
            self.assertEqual(judge.judge_solution(request, cache=Path(tmp)), grade)
            judge.judge_solution(request, cache=Path(tmp))
            self.assertEqual(call.call_count, 1)
            judge.judge_solution(dict(request, candidate="different"), cache=Path(tmp))
            self.assertEqual(call.call_count, 2)
            call.side_effect = RuntimeError("not supported")
            with self.assertRaises(RuntimeError):
                judge.judge_solution(
                    dict(request, candidate="failure"), cache=Path(tmp)
                )
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 2)

    def test_malformed_grades_fail(self):
        for value in [
            {"valid": "false", "elegant": False, "reason": "x"},
            {"valid": False, "elegant": True, "reason": "x"},
            {"valid": True, "elegant": True, "reason": ""},
        ]:
            with self.assertRaises(RuntimeError):
                judge.validate_grade(value)


if __name__ == "__main__":
    unittest.main()
