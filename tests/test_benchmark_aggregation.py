"""Check the paper's benchmark weighting with controlled scores."""
from collections import Counter
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import common
import legibility


def pass_metrics(scores, ks=(1, 4)):
    problems = [{"set": name} for name, _ in scores]
    counts = [correct for _, correct in scores]
    return common.metrics_by_set(problems, counts, n=4, ks=ks)


class BenchmarkAggregation(unittest.TestCase):
    def test_default_loads_the_four_paper_benchmarks(self):
        problems = common.load_problems()
        counts = Counter(p["set"] for p in problems)
        self.assertEqual(counts, {"aime24": 30, "aime25": 30, "aime26": 30,
                                  "amc23_25": 121, "minerva": 272, "olympiad": 581})

    def test_solo_reports_pass_32_without_extrapolating_small_runs(self):
        result = common.agg_pass_at_k([0, 1, 32], n=32)
        self.assertEqual(tuple(result), tuple(f"pass@{k}" for k in (1, 2, 4, 8, 16, 32)))
        self.assertAlmostEqual(result["pass@32"], 2 / 3)
        self.assertAlmostEqual(result["pass@16"], 0.5)
        self.assertNotIn("pass@16", common.agg_pass_at_k([0, 8], n=8))

    def test_aime_contributes_one_quarter_of_the_default_macro(self):
        result = pass_metrics([("aime24", 4), ("aime25", 0), ("aime26", 0),
                               ("amc23_25", 4), ("minerva", 4), ("olympiad", 4)])
        self.assertEqual(len(result["by_benchmark"]), 4)
        self.assertAlmostEqual(result["by_benchmark"]["aime24_26"]["pass@4"], 1 / 3)
        self.assertAlmostEqual(result["macro"]["pass@4"], 5 / 6)
        self.assertEqual(result["aime24"]["pass@4"], 1)
        self.assertEqual(result["aime25"]["pass@4"], 0)

    def test_aime_pools_problems_when_year_sizes_differ(self):
        result = pass_metrics([("aime24", 0), ("aime24", 0), ("aime25", 4),
                               ("aime26", 4), ("amc23_25", 0)])
        self.assertAlmostEqual(result["by_benchmark"]["aime24_26"]["pass@4"], 0.5)
        self.assertAlmostEqual(result["macro"]["pass@4"], 0.25)

    def test_pass_at_k_is_computed_per_problem_before_pooling(self):
        result = pass_metrics([("aime24", 1), ("aime25", 0), ("aime26", 4),
                               ("amc23_25", 0)], ks=(2, 4))
        self.assertAlmostEqual(result["by_benchmark"]["aime24_26"]["pass@2"], 0.5)
        self.assertAlmostEqual(result["by_benchmark"]["aime24_26"]["pass@4"], 2 / 3)
        self.assertAlmostEqual(result["macro"]["pass@2"], 0.25)
        self.assertAlmostEqual(result["macro"]["pass@4"], 1 / 3)

    def test_custom_benchmark_selection_keeps_equal_group_weights(self):
        result = pass_metrics([("math500", 4), ("amc23_25", 0)])
        self.assertEqual(set(result["by_benchmark"]), {"math500", "amc23_25"})
        self.assertEqual(result["macro"]["pass@4"], 0.5)

    def test_legibility_uses_the_same_four_benchmark_groups(self):
        rows = [{"set": name, "ce": value} for name, value in
                [("aime24", 1), ("aime25", 4), ("aime26", 7),
                 ("amc23_25", 0), ("minerva", 0), ("olympiad", 0)]]
        result = legibility.aggregate_ce(rows)
        self.assertEqual(result["by_benchmark"]["aime24_26"], 4)
        self.assertEqual(result["macro_ce"], 1)
        self.assertEqual(result["mean_ce"], 2)
        self.assertEqual(result["by_set"]["aime25"], 4)

    def test_legibility_pools_samples_before_averaging_benchmarks(self):
        rows = [{"set": name, "ce": value} for name, value in
                [("aime24", 0), ("aime24", 0), ("aime25", 3),
                 ("aime26", 3), ("amc23_25", 0)]]
        result = legibility.aggregate_ce(rows)
        self.assertEqual(result["by_benchmark"]["aime24_26"], 1.5)
        self.assertEqual(result["macro_ce"], 0.75)

    def test_empty_legibility_results_remain_unscored(self):
        result = legibility.aggregate_ce([])
        self.assertEqual(result["by_benchmark"], {})
        self.assertIsNone(result["macro_ce"])
        self.assertIsNone(result["mean_ce"])


if __name__ == "__main__":
    unittest.main()
