# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Snapshot review coverage, checked pairs, and conservative whole-method folds."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

try:
    from .review import load_reviews, review_blockers
    from .serve import AuditStore
except ImportError:
    from review import load_reviews, review_blockers
    from serve import AuditStore

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/method-audit")
    args = parser.parse_args()
    report, export, folds = snapshot_reviews(args.data)
    destination = args.data / "machine-review"
    for name, value in [
        ("summary", report),
        ("checked-pairs", export),
        ("holdout-folds", folds),
    ]:
        path = destination / f"{name}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    print(json.dumps(report["coverage"], indent=2))


def snapshot_reviews(data: Path) -> tuple[dict, dict, dict]:
    """Read all current records and snapshot eligible pairs plus independent folds.

    Updates only generated snapshots and, via AuditStore, adds newly defined labels
    to saved partition defaults. It never assigns or changes a problem verdict.
    Missing or blocked reviews remain explicitly counted. Fold membership is based
    on audited mechanisms, not regex proposals or empirical model performance.
    """
    store = AuditStore(data)
    reviews = load_reviews(data, store.problems)
    export = store.machine_export()
    taxonomy = json.loads((data / "machine-review/taxonomy.json").read_text())
    expected = {p["id"] for p in store.problems.values() if p["shortlist"]}
    pairs = [row for partition in export["partitions"].values() for row in partition]
    counts = Counter(reviews[i]["status"] for i in expected & reviews.keys())
    primary = Counter(row["machine_review"]["elegant_method"]["id"] for row in pairs)
    blocked = {
        i: review_blockers(store.problems[i], r, set(store.methods))
        for i, r in reviews.items()
        if r["status"] == "accepted"
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog_sha256": store.catalog_sha256,
        "coverage": {
            "expected": len(expected),
            "reviewed": len(expected & reviews.keys()),
            "unreviewed": len(expected - reviews.keys()),
            "source_routes_expected": sum(
                len(store.problems[i]["routes"]) for i in expected
            ),
            "source_routes_assessed": sum(
                len({a["route_id"] for a in reviews[i]["route_assessments"]})
                for i in expected & reviews.keys()
            ),
            "verdicts": dict(counts),
            "checked_pairs": len(pairs),
            "shortcut_labels": len(primary),
            "all_screened_reviewed": expected <= reviews.keys(),
        },
        "unreviewed_ids": sorted(expected - reviews.keys()),
        "accepted_but_withheld": {
            i: reasons for i, reasons in blocked.items() if reasons
        },
        "shortcut_coverage": [
            {
                "id": m["id"],
                "label": m["label"],
                "classification": m["classification"],
                "definition": m["definition"],
                "checked_pairs": primary[m["id"]],
            }
            for m in taxonomy["methods"]
        ],
        "limits": [
            "Exactly two retained references, not exactly two possible proofs.",
            "Executable checks corroborate answers; written proof validity is model-reviewed.",
            "Junior findability and senior/junior capability gaps are not empirically tested.",
            "Vocabulary size is not a count of independent mechanisms.",
        ],
    }
    folds = build_holdout_folds(pairs, taxonomy)
    report["coverage"]["defined_strategy_holdouts"] = len(folds["folds"])
    report["coverage"]["holdouts_with_at_least_five_test_pairs"] = sum(
        fold["counts"]["test"] >= 5 for fold in folds["folds"]
    )
    folds.update(
        catalog_sha256=store.catalog_sha256,
        taxonomy_sha256=hashlib.sha256(
            (data / "machine-review/taxonomy.json").read_bytes()
        ).hexdigest(),
        created_at=report["created_at"],
    )
    return report, export, folds


def build_holdout_folds(pairs: list[dict], taxonomy: dict) -> dict:
    """Build separate leave-one-mechanism-out folds, excluding immediate neighbors.

    Only precisely defined strategy families become targets. Test requires that
    target as the elegant mechanism. Training excludes its full immediate overlap
    neighborhoods in every reviewed route, including unused alternatives. Folds
    are independent experiments, not partitions to union together. No transitive
    closure or claim of pretraining exclusion is made.
    """
    definitions = {m["id"]: m for m in taxonomy["methods"]}
    groups = {g["id"]: set(g["members"]) for g in taxonomy["related_groups"]}
    for method in definitions.values():
        for group in method.get("related_holdout_groups", []):
            groups.setdefault(group, set()).add(method["id"])
    methods_by_problem = {}
    for row in pairs:
        review = row["machine_review"]
        tags = {m for route in review["route_assessments"] for m in route["methods"]}
        tags.update(
            review[role]["id"] for role in ["elegant_method", "elementary_method"]
        )
        methods_by_problem[row["problem"]["id"]] = tags
    folds = []
    for target, definition in definitions.items():
        if definition["classification"] != "strategy_family" or not definition.get(
            "holdout_eligible"
        ):
            continue
        test = {
            r["problem"]["id"]
            for r in pairs
            if r["machine_review"]["elegant_method"]["id"] == target
        }
        if not test:
            continue
        excluded_methods = {target}
        for group in groups.values():
            if target in group:
                excluded_methods.update(group)
        train = {
            i
            for i, methods in methods_by_problem.items()
            if not methods & excluded_methods
        }
        folds.append(
            {
                "target": target,
                "label": definition["label"],
                "mechanism": definition["definition"],
                "excluded_training_methods": sorted(excluded_methods),
                "train_ids": sorted(train),
                "test_ids": sorted(test),
                "excluded_neighbor_ids": sorted(
                    methods_by_problem.keys() - train - test
                ),
                "counts": {"train": len(train), "test": len(test)},
            }
        )
    return {
        "scope": "Separate conservative leave-one-method-out folds over checked pairs.",
        "status": "Audited-label exclusion verified; semantic near-duplicates and model capability still require calibration.",
        "folds": folds,
    }


if __name__ == "__main__":
    main()
