"""Audit persistence, evidence preservation, and whole-method leakage controls."""

import json
from pathlib import Path
import tempfile
import unittest
from experiments.method_audit.curate import add_problem, load_methods, propose_methods
from experiments.method_audit.serve import AuditStore, Conflict


class MethodAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.problem = {
            "id": "p",
            "title": "Example",
            "question": "Count arrangements",
            "answer": "3",
            "subject": "Counting",
            "sources": [],
            "flags": [],
            "priority": 0,
            "shortlist": True,
            "routes": [
                {
                    "id": "a",
                    "title": "Shortcut",
                    "text": "By a bijection",
                    "kind": "human",
                    "methods": ["bijection"],
                },
                {
                    "id": "b",
                    "title": "Cases",
                    "text": "Enumerate cases",
                    "kind": "human",
                    "methods": ["casework"],
                },
                {
                    "id": "c",
                    "title": "Alternative",
                    "text": "Use symmetry",
                    "kind": "human",
                    "methods": ["symmetry"],
                },
            ],
        }
        catalog = {
            "problems": [self.problem],
            "methods": [
                {"id": "bijection", "split": "train"},
                {"id": "casework", "split": "train"},
                {"id": "symmetry", "split": "test"},
            ],
            "sources": [],
            "notes": [],
            "summary": {},
        }
        (self.root / "catalog.json").write_text(json.dumps(catalog))
        self.store = AuditStore(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_machine_reviews_never_change_human_audits(self):
        folder = self.root / "machine-review/reviews"
        folder.mkdir(parents=True)
        (folder / "p.json").write_text(
            json.dumps(
                {
                    "problem_id": "p",
                    "status": "accepted",
                    "elegant_route_id": "a",
                    "elementary_route_id": "b",
                    "route_assessments": [],
                }
            )
        )
        reviews = self.store.machine_reviews()
        self.assertEqual(reviews["p"]["status"], "accepted")
        self.assertTrue(reviews["p"]["export_blockers"])
        self.assertEqual(self.store.get_audit("p")["status"], "pending")
        self.assertFalse(any(self.store.machine_export()["partitions"].values()))
        self.assertFalse(any(self.store.export()["partitions"].values()))

    def test_machine_records_refresh_without_server_restart(self):
        self.assertEqual(self.store.machine_reviews(), {})
        folder = self.root / "machine-review/reviews"
        folder.mkdir(parents=True)
        (folder / "p.json").write_text(
            json.dumps({"problem_id": "p", "status": "rejected"})
        )
        self.assertEqual(self.store.machine_reviews()["p"]["status"], "rejected")

    def approved(self):
        a = self.store.get_audit("p")
        a.update(
            status="accepted",
            elegant="a",
            elementary="b",
            correct_elegant=True,
            correct_elementary=True,
            distinct=True,
            methods_reviewed=True,
        )
        return a

    def test_unused_alternative_excludes_training(self):
        self.store.save_audit("p", self.approved())
        result = self.store.export()
        self.assertEqual(len(result["partitions"]["test"]), 1)
        self.assertEqual(result["partitions"]["train"], [])

    def test_persistence_and_stale_write(self):
        a = self.approved()
        saved = self.store.save_audit("p", a)
        with self.assertRaises(Conflict):
            self.store.save_audit("p", a)
        self.assertEqual(AuditStore(self.root).get_audit("p"), saved)

    def test_unknown_methods_and_missing_answer_block_export(self):
        a = self.approved()
        a["methods"] = {"c": []}
        a["answer"] = ""
        self.store.problems["p"]["answer"] = ""
        self.store.save_audit("p", a)
        result = self.store.export()
        self.assertFalse(any(result["partitions"].values()))
        self.assertIn(
            "Missing standalone answer", result["blocked_accepted"][0]["reasons"]
        )

    def test_summary_requires_full_replacement(self):
        self.store.problems["p"]["routes"][0]["kind"] = "summary"
        a = self.approved()
        self.store.save_audit("p", a)
        self.assertTrue(self.store.export()["blocked_accepted"])
        a = self.store.get_audit("p")
        a["route_text"] = {"a": "Full independently reviewed derivation."}
        self.store.save_audit("p", a)
        self.assertEqual(len(self.store.export()["partitions"]["test"]), 1)

    def test_source_change_requires_review_again(self):
        self.store.save_audit("p", self.approved())
        self.store.problems["p"]["routes"][0]["text"] = "Changed source"
        self.assertTrue(self.store.export()["blocked_accepted"])

    def test_invalid_method_rejected(self):
        a = self.approved()
        a["additional_methods"] = ["invented"]
        with self.assertRaises(ValueError):
            self.store.save_audit("p", a)

    def test_manual_labels_can_correct_proposals(self):
        a = self.approved()
        a["methods"] = {"c": ["casework"]}
        self.store.save_audit("p", a)
        self.assertEqual(len(self.store.export()["partitions"]["train"]), 1)

    def test_holdout_settings_persist_and_revision_check(self):
        settings = self.store.settings()
        settings["splits"]["symmetry"] = "validation"
        self.store.save_settings(settings)
        with self.assertRaises(Conflict):
            self.store.save_settings(settings)
        self.store.save_audit("p", self.approved())
        self.assertEqual(len(self.store.export()["partitions"]["validation"]), 1)

    def test_exact_dedup_preserves_all_distinct_routes(self):
        cat = {}
        methods = load_methods()
        add_problem(
            cat,
            "a",
            "What is $x$?",
            "2",
            "algebra",
            {"dataset": "a"},
            [("one", "Use Vieta", "human")],
            methods,
        )
        add_problem(
            cat,
            "b",
            "What  is $x$?",
            "2",
            "algebra",
            {"dataset": "b"},
            [("two", "Use cases", "human")],
            methods,
        )
        self.assertEqual(len(cat), 1)
        self.assertEqual(len(next(iter(cat.values())).routes), 2)

    def test_common_language_differentiate_is_not_calculus(self):
        tags, _ = propose_methods(
            "You could differentiate the tall oak trees from the short oak trees.",
            load_methods(),
        )
        self.assertNotIn("calculus", tags)

    def test_added_method_preserves_saved_partition_choices(self):
        settings = self.store.settings()
        settings["splits"]["symmetry"] = "validation"
        self.store.save_settings(settings)
        catalog = json.loads((self.root / "catalog.json").read_text())
        catalog["methods"].append({"id": "new_family", "split": "test"})
        (self.root / "catalog.json").write_text(json.dumps(catalog))
        settings = AuditStore(self.root).settings()
        self.assertEqual(settings["splits"]["symmetry"], "validation")
        self.assertEqual(settings["splits"]["new_family"], "test")

    def test_running_store_exports_its_loaded_catalog_hash(self):
        expected = self.store.catalog_sha256
        (self.root / "catalog.json").write_text("{}")
        self.assertEqual(self.store.export()["catalog_sha256"], expected)


if __name__ == "__main__":
    unittest.main()
