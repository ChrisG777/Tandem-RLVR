"""Load independent all-route elegance judgments without changing proof audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RUBRIC_VERSION = "elegance-v1"


def export_dataset(root: Path, problems: dict, reviews: dict) -> dict:
    """Export all labeled paths of checked problems; refuse incomplete annotation.

    A problem enters the training-eligible subset only if it has at least one
    valid elegant route and one valid non-elegant route. Rejected style contrasts
    stay in the published audit dataset with an explicit exclusion reason.
    """
    rows = []
    missing = []
    for pid, review in reviews.items():
        if review.get("status") != "accepted":
            continue
        if review.get("export_blockers", ["Missing proof/check gate evidence"]):
            raise ValueError(
                f"Accepted problem {pid} has unresolved proof/check blockers"
            )
        problem = problems[pid]
        judgment = load_judgment(root, problem, review)
        if judgment is None:
            missing.append(pid)
            continue
        routes = labeled_routes(problem, review, judgment)
        valid = [r for r in routes if r["correctness"] == "valid"]
        positive = any(r["elegance"]["elegant"] for r in valid)
        negative = any(not r["elegance"]["elegant"] for r in valid)
        reasons = []
        if not positive:
            reasons.append("No valid route was judged elegant")
        if not negative:
            reasons.append("No valid route was judged non-elegant")
        rows.append(
            {
                "problem": dict(problem, routes=routes),
                "machine_review": review,
                "elegance_review": judgment,
                "training_eligible": not reasons,
                "ineligibility_reasons": reasons,
            }
        )
    if missing:
        raise ValueError(
            f"All-route labeling is incomplete: {len(missing)} problems remaining"
        )
    rows.sort(key=lambda r: r["problem"]["id"])
    routes = [route for row in rows for route in row["problem"]["routes"]]
    return {
        "schema_version": 2,
        "rubric_version": RUBRIC_VERSION,
        "scope": "All effective supplied routes of mathematically accepted problems",
        "length_policy": "Shortness is supporting evidence, never a direct reward or sufficient condition",
        "summary": {
            "problems": len(rows),
            "routes": len(routes),
            "elegant_routes": sum(r["elegance"]["elegant"] for r in routes),
            "non_elegant_routes": sum(not r["elegance"]["elegant"] for r in routes),
            "training_eligible_problems": sum(r["training_eligible"] for r in rows),
        },
        "problems": rows,
    }


def judgment_input(problem: dict, review: dict) -> dict:
    """Return the exact question and effective reference texts the judge labels.

    Previously selected roles are intentionally absent. Correctness remains a
    separate prerequisite: an invalid or uncertain reference cannot earn a bonus.
    The fingerprint changes when any text or correctness assessment changes.
    """
    assessments = {r["route_id"]: r for r in review["route_assessments"]}
    value = {
        "problem_id": problem["id"],
        "question": problem["question"],
        "answer": problem["answer"],
        "routes": [
            {
                "route_id": r["id"],
                "text": review.get("route_text", {}).get(r["id"], r["text"]),
                "correctness": assessments[r["id"]]["correctness"],
            }
            for r in problem["routes"]
        ],
    }
    value["input_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return value


def load_judgment(root: Path, problem: dict, review: dict) -> dict | None:
    """Load a complete, current all-route judgment; return None if absent.

    Malformed or stale records raise ValueError rather than silently supplying
    negative labels. This function reads only and never changes mathematical audits.
    """
    path = root / "elegance/reviews" / f"{problem['id']}.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    source = judgment_input(problem, review)
    validate_judgment(value, source)
    return value


def validate_judgment(value: dict, source: dict) -> None:
    """Require one explicit binary label and a reason for every effective route."""
    if (
        value.get("problem_id") != source["problem_id"]
        or value.get("input_sha256") != source["input_sha256"]
        or value.get("rubric_version") != RUBRIC_VERSION
        or value.get("reviewer_model") != "gpt-6.1-sol"
        or not value.get("reviewed_at")
    ):
        raise ValueError("Stale or unidentified elegance judgment")
    labels = value.get("route_labels", [])
    expected = {r["route_id"]: r for r in source["routes"]}
    if len(labels) != len(expected) or {r.get("route_id") for r in labels} != set(
        expected
    ):
        raise ValueError("Elegance judgment must cover every route exactly once")
    for label in labels:
        if (
            type(label.get("elegant")) is not bool
            or not isinstance(label.get("reason"), str)
            or not label["reason"].strip()
            or label.get("shortness_role")
            not in {"supporting", "neutral", "misleading"}
            or label.get("confidence") not in {"high", "medium", "low"}
        ):
            raise ValueError("Incomplete route elegance label")
        if label["elegant"] and expected[label["route_id"]]["correctness"] != "valid":
            raise ValueError(
                "Invalid or uncertain reference cannot be elegance-positive"
            )


def labeled_routes(problem: dict, review: dict, judgment: dict) -> list[dict]:
    """Return all source paths with corrected text, proof status and style labels.

    Original text and provenance remain available. Word counts are descriptive,
    not a threshold or reward, and are computed from the effective reference.
    """
    source = judgment_input(problem, review)
    validate_judgment(judgment, source)
    labels = {r["route_id"]: r for r in judgment["route_labels"]}
    assessments = {r["route_id"]: r for r in review["route_assessments"]}
    result = []
    for original, effective in zip(problem["routes"], source["routes"], strict=True):
        row = dict(original)
        if effective["text"] != original["text"]:
            row["original_text"] = original["text"]
            row["correction_note"] = review.get("correction_notes", {}).get(
                original["id"], review.get("correction_note", "")
            )
        row.update(
            text=effective["text"],
            correctness=effective["correctness"],
            methods=assessments[original["id"]]["methods"],
            elegance=labels[original["id"]],
            word_count=len(effective["text"].split()),
        )
        result.append(row)
    return result
