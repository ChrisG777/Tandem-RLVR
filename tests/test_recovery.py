"""Answer-region and authorship features for the error-recovery analyses."""
import unittest

import numpy as np

from experiments.recovery.authorship import answer_span, sample_features, within_problem_effects


class AnswerRegions(unittest.TestCase):
    def test_last_python_block_and_last_boxed_answer(self):
        text = b"plan\n```python\nx=1\n```\nfix\n```python\nprint(2)\n```\n"
        start, end = answer_span(text, "apps")
        self.assertEqual(text[start:end], b"print(2)\n")
        math = b"so \\boxed{1} wait \\boxed{\\frac{1}{2}} done"
        start, end = answer_span(math, "math")
        self.assertEqual(math[start:end], b"\\boxed{\\frac{1}{2}}")
        self.assertIsNone(answer_span(b"no code", "apps"))

    def test_features_split_junior_tokens_by_region(self):
        pieces = {0: b"plan\n", 1: b"```python\n", 2: b"print(", 3: b"2)\n", 4: b"```\n"}
        features = sample_features([0, 1, 2, 3, 4], [1, 1, 0, 1, 1], pieces, "apps")
        self.assertEqual(features["answer_tokens"], 2)
        self.assertEqual(features["junior_frac_answer"], 0.5)
        self.assertEqual(features["junior_frac_reasoning"], 0.0)
        self.assertEqual(features["switches_per_100_answer"], 100.0)

    def test_within_problem_effect_ignores_problem_difficulty(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(200), 8)
        x = rng.normal(size=(len(groups), 4))
        difficulty = np.repeat(rng.normal(size=200), 8)
        y = (difficulty + 0.5 * x[:, 0] + rng.normal(scale=0.1, size=len(groups)) > 0).astype(float)
        effects = within_problem_effects(x, y, groups, bootstrap=20, seed=0)
        self.assertGreater(effects["junior_frac_answer"]["pp_per_sd"], 5)
        self.assertLess(abs(effects["switches_per_100_reasoning"]["pp_per_sd"]), 3)


if __name__ == "__main__":
    unittest.main()
