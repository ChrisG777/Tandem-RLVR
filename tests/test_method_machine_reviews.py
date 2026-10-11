"""Machine approval cannot substitute for independently executed evidence."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.method_audit.review import (
    load_reviews,
    machine_export,
    source_fingerprint,
)


class MachineReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.p = dict(
            id="p",
            question="Example",
            answer="3",
            routes=[
                dict(id="a", text="short", methods=["short"]),
                dict(id="b", text="long", methods=["long"]),
                dict(id="c", text="alternate", methods=["long"]),
            ],
        )
        self.r = dict(
            problem_id="p",
            status="accepted",
            source_fingerprint=source_fingerprint(self.p),
            elegant_route_id="a",
            elementary_route_id="b",
            route_assessments=[
                dict(route_id=x, correctness="valid", methods=[m])
                for x, m in [("a", "short"), ("b", "long"), ("c", "heldout")]
            ],
            elegant_method=dict(id="short", description="shortcut"),
            elementary_method=dict(id="long", description="routine"),
            contrast_reason="Substantive contrast",
            naturalness_reason="Natural initiation",
            verification=dict(execution_status="passed"),
            route_text={"a": "corrected short"},
            correction_notes={"a": "Corrected typo"},
        )
        self.folder = self.root / "machine-review/reviews"
        self.folder.mkdir(parents=True)
        self.vfolder = self.root / "machine-review/verification"
        self.vfolder.mkdir()
        self.write_review()
        self.splits = dict(short="train", long="train", heldout="test")

    def tearDown(self):
        self.tmp.cleanup()

    def write_review(self):
        self.path = self.folder / "p.json"
        self.path.write_text(json.dumps(self.r))

    def write_evidence(self, **overrides):
        evidence = dict(
            review_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            source_fingerprint=self.r["source_fingerprint"],
            status="passed",
            answer_matches_source=True,
            output="computed 3",
        )
        evidence.update(overrides)
        (self.vfolder / "p.json").write_text(json.dumps(evidence))

    def test_model_cannot_self_assert_execution(self):
        self.assertEqual(
            load_reviews(self.root, [self.p])["p"]["verification"]["execution_status"],
            "unexecuted",
        )
        self.assertFalse(
            any(machine_export(self.root, [self.p], self.splits)["partitions"].values())
        )

    def test_answer_mismatch_and_stale_review_withheld(self):
        self.write_evidence(answer_matches_source=False)
        self.assertFalse(
            any(machine_export(self.root, [self.p], self.splits)["partitions"].values())
        )
        self.write_evidence()
        self.r["contrast_reason"] += " changed"
        self.write_review()
        self.assertFalse(
            any(machine_export(self.root, [self.p], self.splits)["partitions"].values())
        )

    def test_exact_pair_preserves_corrections_and_unused_holdout(self):
        self.write_evidence()
        result = machine_export(self.root, [self.p], self.splits)
        self.assertEqual(len(result["partitions"]["test"]), 1)
        self.assertFalse(result["partitions"]["train"])
        routes = result["partitions"]["test"][0]["problem"]["routes"]
        self.assertEqual([r["id"] for r in routes], ["a", "b"])
        self.assertEqual(routes[0]["text"], "corrected short")
        self.assertEqual(routes[0]["original_text"], "short")
        self.assertEqual(routes[0]["correction_note"], "Corrected typo")

    def test_empty_reviewed_method_labels_block_export(self):
        self.r["route_assessments"][2]["methods"] = []
        self.write_review()
        self.write_evidence()
        self.assertFalse(
            any(machine_export(self.root, [self.p], self.splits)["partitions"].values())
        )


if __name__ == "__main__":
    unittest.main()
