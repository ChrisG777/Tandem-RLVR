"""All-route style labels cannot silently omit paths or bypass proof status."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.method_audit.elegance import (
    export_dataset,
    judgment_input,
    labeled_routes,
    validate_judgment,
)


class EleganceTests(unittest.TestCase):
    def setUp(self):
        self.problem = dict(
            id="p",
            question="Question",
            answer="3",
            routes=[dict(id=key, text=key, methods=[key]) for key in "abc"],
        )
        self.review = dict(
            status="accepted",
            export_blockers=[],
            route_assessments=[
                dict(route_id=key, correctness="valid", methods=[key]) for key in "abc"
            ],
            route_text={"a": "Complete corrected proof"},
            correction_notes={"a": "Fixed omission"},
        )
        self.source = judgment_input(self.problem, self.review)
        self.judgment = dict(
            problem_id="p",
            input_sha256=self.source["input_sha256"],
            rubric_version="elegance-v1",
            reviewer_model="gpt-6.1-sol",
            reviewed_at="2026-10-11",
            route_labels=[
                dict(
                    route_id=key,
                    elegant=key != "b",
                    reason="Specific reason",
                    shortness_role="neutral",
                    confidence="high",
                )
                for key in "abc"
            ],
        )

    def test_multiple_positive_paths_and_originals_preserved(self):
        routes = labeled_routes(self.problem, self.review, self.judgment)
        self.assertEqual(len(routes), 3)
        self.assertEqual(sum(r["elegance"]["elegant"] for r in routes), 2)
        self.assertEqual(routes[0]["original_text"], "a")
        self.assertEqual(routes[0]["text"], "Complete corrected proof")

    def test_missing_duplicate_and_stale_paths_rejected(self):
        for mutation in (
            lambda j: j["route_labels"].pop(),
            lambda j: j["route_labels"].append(j["route_labels"][0]),
            lambda j: j.update(input_sha256="stale"),
        ):
            value = deepcopy(self.judgment)
            mutation(value)
            with self.assertRaises(ValueError):
                validate_judgment(value, self.source)
        self.review["route_text"]["a"] += " changed"
        with self.assertRaises(ValueError):
            validate_judgment(self.judgment, judgment_input(self.problem, self.review))

    def test_incorrect_short_path_cannot_be_positive(self):
        self.review["route_assessments"][0]["correctness"] = "invalid"
        source = judgment_input(self.problem, self.review)
        self.judgment["input_sha256"] = source["input_sha256"]
        with self.assertRaises(ValueError):
            validate_judgment(self.judgment, source)

    def test_export_requires_complete_labels_and_contrast_for_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                export_dataset(root, {"p": self.problem}, {"p": self.review})
            folder = root / "elegance/reviews"
            folder.mkdir(parents=True)
            for row in self.judgment["route_labels"]:
                row["elegant"] = True
            (folder / "p.json").write_text(json.dumps(self.judgment))
            result = export_dataset(root, {"p": self.problem}, {"p": self.review})
            self.assertEqual(result["summary"]["routes"], 3)
            self.assertFalse(result["problems"][0]["training_eligible"])
            self.review["export_blockers"] = ["Stale verification"]
            with self.assertRaises(ValueError):
                export_dataset(root, {"p": self.problem}, {"p": self.review})


if __name__ == "__main__":
    unittest.main()
