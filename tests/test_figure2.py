"""Check Figure 2's statistical units and rejection of incomplete evaluations."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import figure2


class Figure2Statistics(unittest.TestCase):
    def result(self):
        gens = []
        for name, size in (("amc23_25", 121), ("aime24", 30), ("aime25", 30),
                           ("aime26", 30), ("minerva", 272), ("olympiad", 581)):
            for _ in range(size):
                correct = int(name == "amc23_25")
                gens.append({"set": name, "idx": len(gens), "correct": [correct] * 32})
        return {"phase": "solo", "n": 32, "limit": 0, "gens": gens}

    def test_macro_weights_benchmarks_equally_and_constant_scores_have_zero_se(self):
        curves = figure2.summarize(self.result(), bootstrap=20)
        self.assertEqual(curves["macro"]["mean"], [25.] * 6)
        self.assertEqual(curves["macro"]["se"], [0.] * 6)

    def test_incomplete_and_duplicate_results_are_rejected(self):
        for change in (lambda r: r.update(n=8), lambda r: r.update(limit=10),
                       lambda r: r.update(unfinished_chains=1),
                       lambda r: r["gens"].pop(),
                       lambda r: r["gens"].append(r["gens"][0])):
            result = self.result()
            change(result)
            with self.assertRaises(ValueError):
                figure2.summarize(result, bootstrap=20)


if __name__ == "__main__":
    unittest.main()
