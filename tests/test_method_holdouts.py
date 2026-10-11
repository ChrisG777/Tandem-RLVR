"""Whole-method folds exclude related mechanisms even in unused source routes."""

import unittest
from experiments.method_audit.report import build_holdout_folds


class HoldoutTests(unittest.TestCase):
    def test_unused_route_and_manual_neighbor_excluded(self):
        taxonomy = {
            "methods": [
                {
                    "id": "target",
                    "label": "Target",
                    "definition": "Precise mechanism",
                    "classification": "strategy_family",
                    "holdout_eligible": True,
                },
                {
                    "id": "manual",
                    "related_holdout_groups": ["neighbors"],
                    "classification": "strategy_family",
                    "holdout_eligible": True,
                    "label": "Manual",
                    "definition": "Related refinement",
                },
                {
                    "id": "bash",
                    "classification": "execution_primitive_or_broad_tool",
                    "holdout_eligible": False,
                },
            ],
            "related_groups": [{"id": "neighbors", "members": ["target"]}],
        }

        def pair(pid, elegant, route_tags):
            return {
                "problem": {"id": pid},
                "machine_review": {
                    "elegant_method": {"id": elegant},
                    "elementary_method": {"id": "bash"},
                    "route_assessments": [{"methods": tags} for tags in route_tags],
                },
            }

        pairs = [
            pair("test", "target", [["target"], ["bash"]]),
            pair("hidden-alternative", "bash", [["bash"], ["bash"], ["target"]]),
            pair("neighbor", "bash", [["bash"], ["manual"]]),
            pair("clean", "bash", [["bash"], ["bash"]]),
        ]
        fold = build_holdout_folds(pairs, taxonomy)["folds"][0]
        self.assertEqual(fold["test_ids"], ["test"])
        self.assertEqual(fold["train_ids"], ["clean"])
        self.assertEqual(
            fold["excluded_neighbor_ids"], ["hidden-alternative", "neighbor"]
        )
        self.assertEqual(fold["excluded_training_methods"], ["manual", "target"])
        self.assertEqual(len(build_holdout_folds(pairs, taxonomy)["folds"]), 1)


if __name__ == "__main__":
    unittest.main()
