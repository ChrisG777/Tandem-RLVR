"""Focused checks of independent arithmetic execution and answer comparison."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "method_verify",
    Path(__file__).resolve().parents[1] / "experiments/method_audit/verify.py",
)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class VerificationTests(unittest.TestCase):
    def test_conservative_exact_answer_comparison(self):
        self.assertTrue(verify.answers_equal("7/99", r"$\frac{7}{99}$"))
        self.assertTrue(verify.answers_equal("1260", "$1,260$"))
        self.assertTrue(verify.answers_equal("438", r"$\: 438$"))
        self.assertTrue(verify.answers_equal("0.5", r"$\dfrac{1}{2}$"))
        self.assertFalse(verify.answers_equal("0.3333333", r"$\frac{1}{3}$"))
        self.assertFalse(verify.answers_equal("", ""))

    @unittest.skipUnless(importlib.util.find_spec("sympy"), "uv --with sympy required")
    def test_exact_radicals_and_no_rounded_acceptance(self):
        self.assertTrue(
            verify.answers_equal(
                r"\frac{9(\sqrt{3}+3)}{\pi^2}", r"\frac{9}{\pi^2}(\sqrt{3}+3)"
            )
        )
        self.assertTrue(
            verify.answers_equal(r"\frac{15\sqrt{3}}{32}", r"$\frac {15}{32}\sqrt{3}$")
        )
        self.assertTrue(
            verify.answers_equal(r"\sqrt{3}-\frac{3}{4}", r"$\frac{4\sqrt{3}-3}{4}$")
        )
        self.assertFalse(
            verify.answers_equal("0.9820508075688772", r"$\frac{4\sqrt{3}-3}{4}$")
        )
        self.assertFalse(verify.answers_equal(r"\sqrt{3}", r"\sqrt{2}"))

    def test_runtime_access_rejected(self):
        for code in [
            "import os",
            "open('/tmp/a')",
            "x.__class__",
            "from math import *",
        ]:
            with self.assertRaises(ValueError):
                verify.validate_checker(code)

    @unittest.skipUnless(
        Path("/usr/bin/sandbox-exec").exists(), "macOS sandbox required"
    )
    def test_real_enumeration_and_external_answer_check(self):
        code = """from itertools import combinations
from fractions import Fraction
import json
choices = list(combinations(range(12), 5))
good = sum(all(b-a > 1 for a,b in zip(c,c[1:])) for c in choices)
p = Fraction(good, len(choices))
print(json.dumps({'computed_answer': str(p.numerator+p.denominator), 'passed': True}))
"""
        result = verify.execute_checker(code, "$106$")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["computed_answer"], "106")
        mismatch = verify.execute_checker(code, "$107$")
        self.assertEqual(mismatch["status"], "failed", mismatch)
        self.assertFalse(mismatch["answer_matches_source"])


if __name__ == "__main__":
    unittest.main()
