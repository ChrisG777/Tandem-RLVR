"""Exercise interrupted evaluation and incompatible-progress rejection without GPUs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import common
import solo
import handoff
import shorthand


class EvaluationResume(unittest.TestCase):
    def test_interrupted_handoff_reuses_completed_batch(self):
        problems = [{"set": "aime24", "idx": i, "content": str(i), "gt": "1"}
                    for i in range(20)]
        calls = []
        def run_rounds(engines, chains):
            calls.append([c["p"]["idx"] for c in chains])
            if len(calls) == 2:
                raise RuntimeError("simulated preemption")
            for chain in chains:
                chain.update(text="1", done=True)
        tok = lambda text: SimpleNamespace(input_ids=[1])
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "handoff.json")
            with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: tok))}), \
                 patch.object(sys, "argv", ["handoff.py", "--senior", "base", "--junior", "base", "--out", out, "--n", "1", "--single-gpu"]), \
                 patch.object(common, "load_problems", return_value=problems), \
                 patch.object(common, "chat_prefix", side_effect=lambda tok, text: text), \
                 patch.object(common, "grade", return_value=1), \
                 patch.object(handoff, "check_vocab", return_value=True), \
                 patch.object(handoff, "build_engines", return_value=[]), \
                 patch.object(handoff, "run_all_rounds", side_effect=run_rounds):
                with self.assertRaisesRegex(RuntimeError, "preemption"):
                    handoff.main()
                handoff.main()
            self.assertEqual(calls[2], list(range(16, 20)))
            result = json.loads(Path(out).read_text())
            self.assertEqual(len(result["gens"]), 20)
            self.assertEqual(result["unfinished_chains"], 0)

    def test_interrupted_solo_reuses_completed_batch(self):
        problems = [{"set": "aime24", "idx": i, "content": str(i), "gt": "1"}
                    for i in range(20)]
        calls = []

        def generate(prompts, params):
            calls.append(list(prompts))
            if len(calls) == 2:
                raise RuntimeError("simulated preemption")
            return [SimpleNamespace(outputs=[SimpleNamespace(text="1")]) for _ in prompts]

        tokenizer = SimpleNamespace(from_pretrained=lambda _: object())
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "solo.json")
            with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=tokenizer)}), \
                 patch.object(sys, "argv", ["solo.py", "--model", "base", "--out", out, "--n", "1"]), \
                 patch.object(common, "load_problems", return_value=problems), \
                 patch.object(common, "chat_prefix", side_effect=lambda tok, text: text), \
                 patch.object(common, "build_engine", return_value=SimpleNamespace(generate=generate)), \
                 patch.object(common, "grade", return_value=1), \
                 patch.object(solo.config, "sampling_params", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "preemption"):
                    solo.main()
                self.assertFalse(Path(out).exists())
                self.assertEqual(len(json.loads(Path(out + ".progress.json").read_text())["gens"]), 16)
                solo.main()
            result = json.loads(Path(out).read_text())
            self.assertEqual(calls[2], [str(i) for i in range(16, 20)])
            self.assertEqual(len(result["gens"]), 20)
            self.assertEqual(result["metrics"]["macro"]["pass@1"], 1)

    def test_interrupted_shorthand_reuses_completed_batch(self):
        import pandas as pd
        rows = [{"data_source": "string_manipulation",
                 "prompt": [{"role": "user", "content": str(i)}],
                 "reward_model": {"ground_truth": "abc"},
                 "extra_info": {"prompt_sha256": str(i)}} for i in range(20)]
        calls = []
        def generate(prompts, params):
            calls.append(list(prompts))
            if len(calls) == 2:
                raise RuntimeError("simulated preemption")
            sample = SimpleNamespace(text="<answer>abc</answer>", token_ids=[1], finish_reason="stop")
            return [SimpleNamespace(outputs=[sample]) for _ in prompts]
        tok = SimpleNamespace(encode=lambda text: [1])
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "test.parquet", Path(tmp) / "result.json"
            pd.DataFrame(rows).to_parquet(data)
            with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=lambda **kw: kw)}), \
                 patch.object(common, "chat_prefix", side_effect=lambda tok, text: text):
                args = ("base", data, out, 8192)
                kwargs = dict(engine=SimpleNamespace(generate=generate), tokenizer=tok, n=1,
                              max_model_len=10240)
                with self.assertRaisesRegex(RuntimeError, "preemption"):
                    shorthand.evaluate(*args, **kwargs)
                result = shorthand.evaluate(*args, **kwargs)
                with self.assertRaisesRegex(ValueError, "different evaluation"):
                    shorthand.evaluate(*args, **{**kwargs, "max_model_len": 12288})
            self.assertEqual(calls[2], [str(i) for i in range(16, 20)])
            self.assertEqual(result["metrics"]["accuracy"], 1)
            self.assertEqual(len(result["generations"]), 20)
            self.assertEqual(result["provenance"]["max_model_len"], 10240)

    def test_changed_model_or_problem_rejects_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "solo.json")
            problems = [{"set": "aime24", "idx": 0, "content": "original"}]
            path, progress = common.load_progress(out, {"model": "a"}, problems)
            common.save_json(path, progress)
            with self.assertRaisesRegex(ValueError, "inputs differ"):
                common.load_progress(out, {"model": "b"}, problems)
            with self.assertRaisesRegex(ValueError, "inputs differ"):
                common.load_progress(out, {"model": "a"}, [{**problems[0], "content": "changed"}])


if __name__ == "__main__":
    unittest.main()
