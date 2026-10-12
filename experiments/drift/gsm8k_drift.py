"""Correctness-only GSM8K reward, retaining first/last-answer diagnostics."""
import re
from decimal import Decimal, InvalidOperation


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict | None = None, **kwargs) -> dict[str, float]:
    """Return binary final-answer reward; never reward length or unusual language.

    Parse complete numeric answers on #### lines across the entire response.
    First-answer accuracy diagnoses learning to stop rather than learning to solve.
    Missing/malformed answers earn zero; numeric formatting is normalized.
    """
    answers = re.findall(r"(?m)^\s*####\s*([-+]?[\d,]+(?:\.\d+)?)\s*$", solution_str)
    def equal(value: str) -> float:
        try:
            return float(Decimal(value.replace(",", "")) == Decimal(str(ground_truth).replace(",", "")))
        except InvalidOperation:
            return 0.0
    last = equal(answers[-1]) if answers else 0.0
    return {"score": last, "acc": last,
            "first_acc": equal(answers[0]) if answers else 0.0,
            "answer_count": float(len(answers)), "format_ok": float(bool(answers))}
