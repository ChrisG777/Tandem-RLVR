"""MedCalc grader, stratified split, and result table. No GPU and no dataset download."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.medcalc.grade import (
    DATE_IDS,
    DECIMAL_IDS,
    INTEGER_IDS,
    WEEKS_IDS,
    extract_boxed,
    score_response,
)
from experiments.medcalc.protocol import holdout_mask, render_prompt
from experiments.medcalc.report import collect, render_table
from experiments.medcalc.reward import compute_score
from experiments.medcalc.rows import make_record


def gold(**overrides):
    payload = {
        "answer": "100",
        "lower": "95",
        "upper": "105",
        "calculator_id": 2,
        "output_type": "decimal",
    }
    payload.update(overrides)
    return json.dumps(payload)


class CalculatorPartition(unittest.TestCase):
    def test_official_ids_partition_fifty_five_calculators(self):
        groups = (DATE_IDS, WEEKS_IDS, INTEGER_IDS, DECIMAL_IDS)
        self.assertEqual(sum(len(group) for group in groups), 55)
        combined = set().union(*groups)
        self.assertEqual(len(combined), 55)
        self.assertIn(69, WEEKS_IDS)
        self.assertNotIn(69, INTEGER_IDS)


class Scoring(unittest.TestCase):
    def test_decimal_band_and_boxed_requirement(self):
        target = gold()
        self.assertEqual(score_response(r"working \boxed{100}", target), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{95}", target), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{105}", target), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{94.9}", target), (0.0, 1.0))
        self.assertEqual(score_response("the answer is 100", target), (0.0, 0.0))
        self.assertEqual(score_response(r"\boxed{90} then \boxed{100}", target), (1.0, 1.0))

    def test_units_percent_and_negative_limits(self):
        self.assertEqual(score_response(r"\boxed{100 mL/min}", gold()), (1.0, 1.0))
        percent = gold(answer="0.53", lower="0.50", upper="0.56")
        self.assertEqual(score_response(r"\boxed{53%}", percent), (1.0, 1.0))
        renal = gold(answer="8.1", lower="7", upper="9")
        self.assertEqual(score_response(r"\boxed{8.1 mL/min/1.73 m2}", renal), (1.0, 1.0))
        negative = gold(answer="-10", lower="-9.5", upper="-10.5")
        self.assertEqual(score_response(r"\boxed{-10}", negative), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{-8}", negative), (0.0, 1.0))

    def test_integer_rules(self):
        target = gold(answer="3", lower="3", upper="3", calculator_id=4, output_type="integer")
        self.assertEqual(score_response(r"\boxed{3}", target), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{3.2}", target), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{3 out of 5}", target), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{3, 4, 5}", target), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{1,200}", gold(
            answer="1200", lower="1200", upper="1200", calculator_id=4, output_type="integer")), (1.0, 1.0))

    def test_dates_and_weeks(self):
        dated = gold(answer="01/02/2024", lower="01/02/2024", upper="01/02/2024",
                     calculator_id=13, output_type="date")
        self.assertEqual(score_response(r"\boxed{1/2/2024}", dated), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{02/31/2024}", dated), (0.0, 1.0))
        weeks = gold(answer="(17 weeks, 4 days)", lower="(17 weeks, 4 days)",
                     upper="(17 weeks, 4 days)", calculator_id=69, output_type="time")
        self.assertEqual(score_response(r"\boxed{(17, 4)}", weeks), (1.0, 1.0))
        self.assertEqual(score_response(r"\boxed{17 weeks, 5 days}", weeks), (0.0, 1.0))

    def test_unknown_calculator_uses_output_type(self):
        target = gold(calculator_id=999, output_type="decimal")
        self.assertEqual(score_response(r"\boxed{100}", target), (1.0, 1.0))

    def test_reward_entry_point(self):
        scored = compute_score("medcalc", r"\boxed{100}", gold())
        self.assertEqual(scored["score"], 1.0)
        self.assertEqual(scored["acc"], 1.0)
        self.assertEqual(scored["answer_present"], 1.0)
        with self.assertRaises(ValueError):
            compute_score("math", r"\boxed{1}", gold())
        with self.assertRaises(ValueError):
            compute_score("medcalc", r"\boxed{1}", "{}")

    def test_boxed_extractor_ignores_prose(self):
        self.assertIsNone(extract_boxed("100"))
        self.assertEqual(extract_boxed(r"\boxed{1} \boxed{2}"), "2")


class Records(unittest.TestCase):
    def test_prompt_hides_the_answer_and_roundtrips_the_reward(self):
        row = {
            "Patient Note": "Age 16. Weight 55 kg.",
            "Question": "What is the creatinine clearance?",
            "Ground Truth Answer": "141.04",
            "Lower Limit": "133.99",
            "Upper Limit": "148.09",
            "Calculator ID": 2,
            "Output Type": "decimal",
            "Category": "lab test",
            "Calculator Name": "Creatinine Clearance",
            "Note ID": "pmc-1",
            "Row Number": 7,
        }
        record = make_record(row, 0)
        content = record["prompt"][0]["content"]
        self.assertIn("Age 16", content)
        self.assertIn("creatinine clearance", content.lower())
        self.assertIn(r"\boxed{}", content)
        self.assertNotIn("141.04", content)
        self.assertEqual(record["data_source"], "medcalc")
        self.assertEqual(record["extra_info"]["task_id"], 7)
        self.assertEqual(record["extra_info"]["category_slug"], "lab_test")
        self.assertEqual(compute_score("medcalc", r"\boxed{141}", record["reward_model"]["ground_truth"])["acc"], 1.0)

    def test_holdout_is_stratified_and_deterministic(self):
        ids = [calculator for calculator in range(10) for _ in range(20)]
        first = holdout_mask(ids, 50, 42)
        second = holdout_mask(ids, 50, 42)
        self.assertEqual(first, second)
        self.assertEqual(sum(first), 50)
        self.assertEqual(len(first), len(ids))
        for calculator in range(10):
            rows = [held for held, calculator_id in zip(first, ids) if calculator_id == calculator]
            self.assertGreaterEqual(rows.count(False), 1)
            self.assertGreaterEqual(rows.count(True), 1)
        self.assertNotEqual(first, holdout_mask(list(reversed(ids)), 50, 42))
        singleton = holdout_mask([1, 1, 1, 2], 1, 0)
        self.assertEqual(sum(singleton), 1)
        self.assertFalse(singleton[3])


class Report(unittest.TestCase):
    def test_retention_uses_macro_pass_at_4(self):
        solo = {"metrics": {"macro": {"pass@1": 0.5, "pass@4": 0.8}, "by_benchmark": {}}}
        handoff = {"metrics": {"macro": {"pass@4": 0.6}, "by_benchmark": {}}}
        summary = {"arms": {"grpo": {
            "solo": {"macro": solo["metrics"]["macro"]},
            "handoff": {"macro": handoff["metrics"]["macro"], "retention_of_solo_pass@4": 0.75},
        }}}
        text = render_table(summary)
        self.assertIn("75.0%", text)
        self.assertIn("grpo", text)
        self.assertIsNone(collect(Path("/this/path/does/not/exist"))["arms"].get("grpo"))


if __name__ == "__main__":
    unittest.main()
