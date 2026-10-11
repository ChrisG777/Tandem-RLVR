"""Read durable model reviews and export pairs that pass independent checks."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path


def source_fingerprint(problem: dict) -> str:
    """Hash immutable question, answer and all supplied routes/labels."""
    value = [
        problem["question"],
        problem["answer"],
        [(r["id"], r["text"], r["methods"]) for r in problem["routes"]],
    ]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_reviews(data_root: Path, problems: list[dict] | dict) -> dict[str, dict]:
    """Read separate model records without modifying human audits."""
    result = {}
    for path in sorted((Path(data_root) / "machine-review" / "reviews").glob("*.json")):
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            if isinstance(value, dict) and value.get("problem_id") == path.stem:
                value.setdefault(
                    "provenance",
                    {
                        "model": value.get(
                            "reviewer_model", value.get("reviewer", "unknown")
                        ),
                        "mode": value.get(
                            "review_kind", "native individual source review"
                        ),
                        "reviewed_at": value.get("reviewed_at"),
                    },
                )
                value.setdefault("verification", {})["execution_status"] = "unexecuted"
                sidecar = (
                    Path(data_root) / "machine-review" / "verification" / path.name
                )
                if sidecar.exists():
                    evidence = json.loads(sidecar.read_text())
                    if evidence.get("review_sha256") == hashlib.sha256(
                        raw
                    ).hexdigest() and evidence.get("source_fingerprint") == value.get(
                        "source_fingerprint"
                    ):
                        value["verification"]["execution_status"] = (
                            "passed"
                            if evidence.get("status") == "passed"
                            and evidence.get("answer_matches_source") is True
                            else (
                                "failed"
                                if evidence.get("status") == "passed"
                                else evidence.get("status", "unexecuted")
                            )
                        )
                        value["verification"]["execution_output"] = evidence.get(
                            "output", ""
                        )
                        value["verification"]["execution_evidence"] = evidence
                result[path.stem] = value
        except (OSError, ValueError):
            continue
    return result


def review_blockers(
    problem: dict, review: dict, method_ids: set[str] | list[str]
) -> list[str]:
    """Explain why a model-selected pair cannot enter a verified export."""
    reasons = []
    if review.get("status") != "accepted":
        reasons.append("Model review is not accepted.")
    if review.get("source_fingerprint") != source_fingerprint(problem):
        reasons.append("Source changed after review.")
    routes = {r["id"] for r in problem["routes"]}
    selected = [review.get("elegant_route_id"), review.get("elementary_route_id")]
    if len(set(selected)) != 2 or any(x not in routes for x in selected):
        reasons.append("Exactly two distinct supplied routes are required.")
    assessments = {r.get("route_id"): r for r in review.get("route_assessments", [])}
    if any(
        not isinstance(row.get("methods"), list)
        or not row.get("methods")
        or any(m not in method_ids for m in row.get("methods", []))
        for row in assessments.values()
    ):
        reasons.append("Every route needs reviewed recognized method labels.")
    if any(
        row.get("correctness") not in {"valid", "invalid", "uncertain"}
        for row in assessments.values()
    ):
        reasons.append("Every route needs a recognized correctness judgment.")
    if set(assessments) != routes:
        reasons.append("Every supplied route must be assessed.")
    if any(assessments.get(x, {}).get("correctness") != "valid" for x in selected):
        reasons.append("Both selected routes must be valid.")
    for role in ["elegant_method", "elementary_method"]:
        m = review.get(role, {})
        if m.get("id") not in method_ids or not m.get("description"):
            reasons.append(f"{role} needs a described recognized method.")
    if not review.get("contrast_reason") or not review.get("naturalness_reason"):
        reasons.append("Contrast and findability require specific reasons.")
    verification = review.get("verification", {})
    if verification.get("execution_status") != "passed":
        reasons.append("Independent executable answer check has not passed.")
    return reasons


def machine_export(
    data_root: Path, problems: list[dict] | dict, splits: dict[str, str]
) -> dict:
    """Export exactly two reviewed paths; all retained methods determine holdout."""
    values = list(problems.values()) if isinstance(problems, dict) else problems
    reviews = load_reviews(data_root, values)
    out = {
        "split_status": "provisional audited-label split; semantic family overlap audit required before training",
        "review_scope": "AI mathematical review plus executable answer corroboration, not empirical junior calibration",
        "partitions": {"train": [], "validation": [], "test": []},
        "withheld": [],
    }
    for p in values:
        r = reviews.get(p["id"])
        if not r:
            continue
        blockers = review_blockers(p, r, set(splits))
        if blockers:
            out["withheld"].append({"problem_id": p["id"], "reasons": blockers})
            continue
        methods = {
            m for route in r["route_assessments"] for m in route.get("methods", [])
        }
        methods.update(r[k]["id"] for k in ["elegant_method", "elementary_method"])
        partition = (
            "test"
            if any(splits.get(m) == "test" for m in methods)
            else "validation"
            if any(splits.get(m) == "validation" for m in methods)
            else "train"
        )
        selected = {r["elegant_route_id"], r["elementary_route_id"]}
        export_routes = []
        for route in p["routes"]:
            if route["id"] not in selected:
                continue
            exported = dict(route, methods=assessments_methods(r, route["id"]))
            replacement = r.get("route_text", {}).get(route["id"], "")
            if replacement:
                exported.update(
                    original_text=route["text"],
                    text=replacement,
                    correction_note=r.get("correction_notes", {}).get(
                        route["id"], r.get("correction_note", "")
                    ),
                )
            export_routes.append(exported)
        out["partitions"][partition].append(
            {"problem": {**p, "routes": export_routes}, "machine_review": r}
        )
    return out


def assessments_methods(review: dict, route_id: str) -> list[str]:
    return next(
        row["methods"]
        for row in review["route_assessments"]
        if row["route_id"] == route_id
    )
