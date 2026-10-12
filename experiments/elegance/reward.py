"""Answer-gated elegance reward for the existing verl custom-reward interface."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.elegance.judge import judge_solution  # noqa: E402
from reward.math_boxed_reward import compute_score as answer_score  # noqa: E402


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict:
    """Return correctness + 0.5 * valid elegant proof; fail on judge errors.

    Requires question/references in extra_info and ELEGANCE_JUDGE_CACHE.
    Incorrect answers skip the external call. Token count never contributes to
    reward. The pinned GPT-6.1-sol judge may recognize unseen valid methods.
    """
    if data_source != "method_elegance":
        raise ValueError(f"Unsupported source: {data_source}")
    base = answer_score(data_source, solution_str, ground_truth, extra_info)
    result = {**base, "valid_derivation": 0.0, "elegant": 0.0, "judge_called": 0.0}
    if not base["acc"]:
        return result
    if not extra_info or "question" not in extra_info or "references" not in extra_info:
        raise ValueError("Missing audited judge context")
    # Parquet-compatible JSON text avoids nested heterogeneous Arrow records.
    import json

    references = extra_info["references"]
    if isinstance(references, str):
        references = json.loads(references)
    request = {
        "question": extra_info["question"],
        "candidate": solution_str,
        "answer": str(ground_truth),
        "references": references,
    }
    grade = judge_solution(
        request,
        cache=Path(os.environ["ELEGANCE_JUDGE_CACHE"]),
        executable=os.environ.get("ELEGANCE_CODEX_BIN", "codex"),
    )
    result.update(
        valid_derivation=float(grade["valid"]),
        elegant=float(grade["elegant"]),
        judge_called=1.0,
        score=base["acc"] + 0.5 * grade["valid"] * grade["elegant"],
    )
    return result
