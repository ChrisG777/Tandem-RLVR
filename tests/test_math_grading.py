"""Regressions for silent backend fallback and equivalent symbolic/numeric answers."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reward'))
import math_boxed_reward as reward
from hendrycks_math_grader import normalize_scientific_literal


class MathGrading(unittest.TestCase):
    def test_equivalent_answers(self):
        for answer, target in [(r'p(s)=s^2+\omega^2', r's^2+\omega^2'),
                               ('b=m', 'm'), (r'Y(s)=\frac{1}{s+a}', r'\frac{1}{s+a}'),
                               (r'2.55\times10^{-10}', '2.55e-10'),
                               ('2.55e-10', r'2.55\times10^{-10}'),
                               ('2E3', '2000'), (r'\frac{1}{2}', '0.5')]:
            with self.subTest(answer=answer):
                self.assertEqual(reward.compute_score('math', r'\boxed{'+answer+'}', target)['acc'], 1)

    def test_incorrect_answers_stay_incorrect(self):
        for answer, target in [('2.55e-9', '2.55e-10'), ('2.55', '2.55e-10'),
                               ('0', '2.55e-10'), ('x=3', '2'), ('-2', '2')]:
            with self.subTest(answer=answer):
                self.assertEqual(reward.compute_score('math', r'\boxed{'+answer+'}', target)['acc'], 0)
        self.assertEqual(reward.compute_score('math', 'unboxed 2', '2')['acc'], 0)

    def test_missing_backend_is_not_a_zero_reward(self):
        with patch.object(reward, '_try_import_math_verify', side_effect=RuntimeError('missing backend')):
            with self.assertRaisesRegex(RuntimeError, 'missing backend'):
                reward.compute_score('math', r'\boxed{1}', '1')

    def test_e_in_expressions_is_not_rewritten(self):
        for expression in ('e', 'e-10', 'x=2e3', '2e3 m', r'e^{x}', '2.55e-10+1'):
            self.assertEqual(normalize_scientific_literal(expression), expression)


if __name__ == '__main__':
    unittest.main()
