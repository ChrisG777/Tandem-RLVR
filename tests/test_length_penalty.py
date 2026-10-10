"""Exercise the real verl reward adapter, including padding and correctness."""
import asyncio
from pathlib import Path
import sys
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch
from verl import DataProto
from verl.trainer.ppo.reward import load_reward_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reward"))


class LengthPenalty(unittest.TestCase):
    def score(self, text, length, task="ruletaker_shared", target="true true true true", coefficient=.2):
        config = OmegaConf.create({
            "data": {"max_response_length": 100},
            "reward": {
                "length_penalty": coefficient,
                "reward_manager": {"source": "importlib", "name": "LengthPenaltyRewardManager",
                                   "module": {"path": str(ROOT / "reward/length_penalty.py")}},
                "custom_reward_function": {"path": str(ROOT / "reward/shorthand_reward.py"),
                                           "name": "compute_score"},
            },
        })
        class Tokenizer:
            def decode(self, ids, **kwargs):
                self.last_length = len(ids)
                return text
        tokenizer = Tokenizer()
        data = DataProto.from_dict(
            tensors={"responses": torch.ones(1, 100, dtype=torch.long),
                     "prompts": torch.ones(1, 7, dtype=torch.long),
                     "attention_mask": torch.tensor([[1] * (7 + length) + [0] * (100 - length)])},
            non_tensors={"data_source": np.array([task], dtype=object),
                         "reward_model": np.array([{"ground_truth": target}], dtype=object),
                         "extra_info": np.array([{}], dtype=object)},
        )
        async def run():
            manager = load_reward_manager(config, tokenizer)
            result = await manager.run_single(data)
            assembled = manager.assemble_rm_scores(data, [result["reward_score"]])
            self.assertAlmostEqual(float(assembled.sum()), result["reward_score"], places=6)
            self.assertEqual(int(torch.count_nonzero(assembled)), int(result["reward_score"] != 0))
            self.assertEqual(tokenizer.last_length, length)
            return result
        return asyncio.run(run())

    def test_padding_prompt_and_correctness_diagnostics(self):
        result = self.score("<answer>true true true true</answer>", 50)
        self.assertAlmostEqual(result["reward_score"], .9)
        self.assertEqual(result["reward_extra_info"]["acc"], 1)
        self.assertEqual(result["reward_extra_info"]["query_accuracy"], 1)
        self.assertEqual(result["reward_extra_info"]["response_tokens"], 50)
        self.assertEqual(result["reward_extra_info"]["correctness_reward"], 1)

    def test_correctness_dominates_length(self):
        full = self.score("<answer>true true true true</answer>", 100)
        partial = self.score("<answer>true true true false</answer>", 1)
        self.assertGreater(full["reward_score"], partial["reward_score"])
        short = self.score("<answer>true true true true</answer>", 10)
        self.assertGreater(short["reward_score"], full["reward_score"])

    def test_matrix_and_malformed_outputs(self):
        correct = self.score("<answer>1 2\n3 4</answer>", 50, "manipulate_matrix", "1 2\n3 4")
        self.assertAlmostEqual(correct["reward_score"], .9)
        malformed = self.score("no final answer", 50)
        self.assertAlmostEqual(malformed["reward_score"], -.1)
        self.assertEqual(malformed["reward_extra_info"]["acc"], 0)

    def test_zero_coefficient_and_invalid_configuration(self):
        result = self.score("<answer>true true true true</answer>", 100, coefficient=0)
        self.assertEqual(result["reward_score"], 1)
        for coefficient in (-.1, .25, float("nan")):
            with self.assertRaises(ValueError):
                self.score("x", 1, coefficient=coefficient)
        with self.assertRaises(ValueError):
            self.score("x", 0)


if __name__ == "__main__":
    unittest.main()
