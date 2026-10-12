# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Package checked, all-path-labeled problems for publication and local browsing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

try:
    from .report import build_holdout_folds
    from .serve import AuditStore
except ImportError:
    from report import build_holdout_folds
    from serve import AuditStore

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/method-audit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_package(args.data, args.out), indent=2))


def write_package(data: Path, destination: Path) -> dict:
    """Atomically write a portable ZIP and adjacent manifest; never upload it.

    Requires complete current elegance labels and passing proof/answer gates.
    Only accepted source problems, machine evidence, taxonomy and local math
    assets are included. Human SQLite audits, credentials and unrelated files
    are never included. Existing destination files are replaced after success.
    """
    store = AuditStore(data)
    dataset = store.elegance_export()
    rows = dataset["problems"]
    eligible = [r for r in rows if r["training_eligible"]]
    taxonomy = json.loads((data / "machine-review/taxonomy.json").read_text())
    folds = build_holdout_folds(eligible, taxonomy)
    # Retain the prior decisive mechanism only where that selected route is
    # independently confirmed elegant by the new all-path judge.
    confirmed_primary = {
        row["problem"]["id"]
        for row in eligible
        if next(
            r
            for r in row["problem"]["routes"]
            if r["id"] == row["machine_review"]["elegant_route_id"]
        )["elegance"]["elegant"]
    }
    for fold in folds["folds"]:
        removed = set(fold["test_ids"]) - confirmed_primary
        fold["test_ids"] = sorted(set(fold["test_ids"]) & confirmed_primary)
        fold["excluded_neighbor_ids"] = sorted(
            set(fold["excluded_neighbor_ids"]) | removed
        )
        fold["counts"]["test"] = len(fold["test_ids"])
    folds["folds"] = [f for f in folds["folds"] if f["test_ids"]]
    folds["elegance_policy"] = (
        "Test requires the previously identified decisive route to be confirmed elegant; train excludes all target/neighbor labels in every supplied route."
    )
    catalog = dict(store.catalog)
    catalog["problems"] = [store.problems[row["problem"]["id"]] for row in rows]
    used_sources = {s["dataset"] for p in catalog["problems"] for s in p["sources"]}
    catalog["sources"] = [s for s in catalog["sources"] if s["name"] in used_sources]
    catalog["summary"] = dict(
        dataset["summary"],
        screened_shortlist=len(rows),
        independently_verified_pairs=len(rows),
    )
    catalog["notes"] = [
        "Published subset of accepted problems, with every supplied path labeled by GPT-6.1-sol.",
        "Proof validity and elegance are distinct judgments. Shortness is supporting evidence only.",
        "Model accessibility and a senior/junior capability gap have not been measured.",
        "Use the conservative holdout-folds.json for experiments; UI label partitions are provisional.",
    ]
    files = {
        "catalog.json": encode(catalog),
        "dataset.json": encode(dataset),
        "holdout-folds.json": encode(folds),
        "machine-review/taxonomy.json": (
            data / "machine-review/taxonomy.json"
        ).read_bytes(),
    }
    for row in rows:
        pid = row["problem"]["id"]
        for directory in [
            "machine-review/reviews",
            "machine-review/verification",
            "elegance/reviews",
        ]:
            name = f"{directory}/{pid}.json"
            files[name] = (data / name).read_bytes()
    for path in sorted((data / "assets/katex").rglob("*")):
        if path.is_file() and (
            path.suffix in {".js", ".css", ".woff2"} or path.name == "LICENSE"
        ):
            files[str(path.relative_to(data))] = path.read_bytes()
    if "assets/katex/LICENSE" not in files:
        raise ValueError("Portable UI requires the pinned KaTeX assets and license")
    valid = [
        r
        for row in rows
        for r in row["problem"]["routes"]
        if r["correctness"] == "valid"
    ]
    summary = dict(
        dataset["summary"],
        defined_strategy_holdouts=len(folds["folds"]),
        holdouts_with_at_least_five_test_pairs=sum(
            f["counts"]["test"] >= 5 for f in folds["folds"]
        ),
        median_words_valid_elegant=statistics.median(
            r["word_count"] for r in valid if r["elegance"]["elegant"]
        ),
        median_words_valid_non_elegant=statistics.median(
            r["word_count"] for r in valid if not r["elegance"]["elegant"]
        ),
    )
    manifest = {
        "schema_version": 2,
        "rubric_version": dataset["rubric_version"],
        "summary": summary,
        "parent_catalog_sha256": store.catalog_sha256,
        "sources": catalog["sources"],
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(files.items())
        },
    }
    files["manifest.json"] = encode(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(
                name
            )  # Fixed timestamp makes the bundle reproducible.
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    temporary.replace(destination)
    manifest["archive_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".manifest.json").write_bytes(encode(manifest))
    return {
        "archive": str(destination),
        "sha256": manifest["archive_sha256"],
        **summary,
    }


def encode(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()


if __name__ == "__main__":
    main()
