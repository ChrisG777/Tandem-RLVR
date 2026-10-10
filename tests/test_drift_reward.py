"""Protect outcome-only reward and the first-vs-last diagnostic."""
import unittest
from reward.gsm8k_drift import compute_score


class DriftRewardTest(unittest.TestCase):
    def test_entire_response_and_final_answer(self):
        r = compute_score("gsm8k", "#### 42\n" + "reasoning " * 100 + "\n#### 7", "42")
        self.assertEqual((r["score"], r["first_acc"], r["answer_count"]), (0, 1, 2))

    def test_no_style_or_length_reward(self):
        for text in ["####42", "Ordinary English.\n#### 42.0", "Zqx " * 1000 + "\n#### +42"]:
            self.assertEqual(compute_score("gsm8k", text, "42")["score"], 1)

    def test_reject_malformed_and_missing(self):
        for text in ["42", "#### 42 bananas", "#### 42.3.4", "#### NaN"]:
            self.assertEqual(compute_score("gsm8k", text, "42")["score"], 0)

    def test_numeric_format(self):
        self.assertEqual(compute_score("gsm8k", "#### -1,200.0", "-1200")["score"], 1)


if __name__ == "__main__":
    unittest.main()
