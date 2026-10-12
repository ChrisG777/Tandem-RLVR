"""verl reward entry point. Binary score on the boxed MedCalc answer."""

import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grade import score_response


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return the paper's binary reward fields for one tandem or GRPO rollout."""
    if str(data_source) != "medcalc":
        raise ValueError(f"Unsupported data source: {data_source}")
    correct, present = score_response(solution_str or "", ground_truth)
    return {"score": correct, "acc": correct, "answer_present": present}


def grader_identity() -> dict:
    paths = [Path(__file__), Path(__file__).with_name("grade.py")]
    return {
        "name": "medcalc-boxed-v1",
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
