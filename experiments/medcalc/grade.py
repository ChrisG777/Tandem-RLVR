"""Score a boxed clinical answer with MedCalc-Bench's calculator rules.

Calculator ids and the decimal band, integer rounding, date, and weeks/days
rules follow ``evaluation/evaluate.py`` in nikhilk7153/MedCalc-Bench-Verified.
Model text is parsed as a literal. It is never passed to ``eval``.
"""

import json
import re
from datetime import date

# Branches in the official checker, with calculator 69 kept on the weeks path
# (that branch runs before the integer branch).
DATE_IDS = frozenset({13, 68})
WEEKS_IDS = frozenset({69})
INTEGER_IDS = frozenset({
    4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51,
})
DECIMAL_IDS = frozenset({
    2, 3, 5, 6, 7, 8, 9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44,
    46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
})

_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_DATE = re.compile(r"(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})")
_WEEKS = re.compile(
    r"\(?\s*[\"']?(\d+)\s*(?:weeks?)?[\"']?\s*,\s*[\"']?(\d+)\s*(?:days?)?[\"']?\s*\)?",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_LISTED = re.compile(r"-?\d+(?:, ?-?\d+)+")
_THOUSANDS = re.compile(r"-?\d{1,3}(?:,\d{3})+")
_OUT_OF = re.compile(r"(\d+)\s+out of\b", re.IGNORECASE)
_ML = re.compile(r"(-?\d+(?:\.\d+)?)\s*mL/min/1\.73")
_PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def extract_boxed(text: str) -> str | None:
    """Return the last ``\\boxed{...}`` span, or None when the response has none."""
    found = list(_BOXED.finditer(text or ""))
    if not found:
        return None
    return found[-1].group(1).strip()


def answer_kind(payload: dict) -> str:
    """Return decimal, integer, date, or weeks for a gold payload."""
    calculator_id = int(payload["calculator_id"])
    if calculator_id in DATE_IDS:
        return "date"
    if calculator_id in WEEKS_IDS:
        return "weeks"
    if calculator_id in INTEGER_IDS:
        return "integer"
    if calculator_id in DECIMAL_IDS:
        return "decimal"
    output = str(payload.get("output_type", "")).strip().lower()
    if output == "decimal":
        return "decimal"
    if output == "integer":
        return "integer"
    if "date" in output:
        return "date"
    if "week" in output or output == "time":
        return "weeks"
    raise ValueError(f"Unknown calculator {calculator_id} with output type {output!r}")


def parse_payload(ground_truth) -> dict:
    if isinstance(ground_truth, dict):
        payload = dict(ground_truth)
    else:
        payload = json.loads(str(ground_truth))
    missing = [key for key in ("answer", "lower", "upper", "calculator_id") if key not in payload]
    if missing:
        raise ValueError(f"MedCalc gold is missing {', '.join(missing)}")
    payload["calculator_id"] = int(payload["calculator_id"])
    return payload


def parse_date(text: str):
    match = _DATE.search(str(text))
    if not match:
        return None
    month, day, year = (int(match.group(index)) for index in (1, 2, 3))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_weeks(text: str):
    match = _WEEKS.search(str(text))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def last_number(text: str):
    numbers = _NUMBER.findall(str(text).replace(",", ""))
    if not numbers:
        return None
    return float(numbers[-1])


def extract_integer(text: str):
    """Official integer extraction: 'N out of', comma-lists, otherwise rounded last number."""
    out_of = _OUT_OF.search(text)
    if out_of:
        return int(out_of.group(1))
    listed = _LISTED.search(text)
    if listed and _THOUSANDS.fullmatch(listed.group(0)) is None:
        return len(listed.group(0).split(","))
    value = last_number(text)
    if value is None:
        return None
    return int(round(value))


def extract_decimal(text: str):
    """Official decimal extraction: indexed renal rate, then a trailing percent, then the last number."""
    renal = _ML.search(text)
    if renal:
        return float(renal.group(1))
    percents = _PERCENT.findall(text)
    if percents:
        return float(percents[-1]) / 100.0
    return last_number(text)


def _interval(payload: dict) -> tuple[float, float]:
    lower = float(str(payload["lower"]).replace(",", ""))
    upper = float(str(payload["upper"]).replace(",", ""))
    if lower > upper:
        return upper, lower
    return lower, upper


def matches(span: str, payload: dict) -> bool:
    kind = answer_kind(payload)
    if kind == "date":
        predicted = parse_date(span)
        gold = parse_date(str(payload["answer"]))
        return predicted is not None and predicted == gold
    if kind == "weeks":
        predicted = parse_weeks(span)
        gold = parse_weeks(str(payload["answer"]))
        return predicted is not None and predicted == gold
    if kind == "integer":
        predicted = extract_integer(span)
        if predicted is None:
            return False
        gold = int(round(float(str(payload["answer"]).replace(",", ""))))
        return predicted == gold
    if kind == "decimal":
        predicted = extract_decimal(span)
        if predicted is None:
            return False
        lower, upper = _interval(payload)
        return lower <= predicted <= upper
    raise ValueError(f"Unhandled answer kind: {kind}")


def validate_gold(payload: dict) -> None:
    """Reject a training row whose gold answer cannot be scored."""
    kind = answer_kind(payload)
    if kind == "date" and parse_date(str(payload["answer"])) is None:
        raise ValueError(f"Unparseable date answer: {payload['answer']!r}")
    if kind == "weeks" and parse_weeks(str(payload["answer"])) is None:
        raise ValueError(f"Unparseable weeks/days answer: {payload['answer']!r}")
    if kind == "integer":
        float(str(payload["answer"]).replace(",", ""))
    if kind == "decimal":
        float(str(payload["answer"]).replace(",", ""))
        _interval(payload)


def score_response(text: str, ground_truth) -> tuple[float, float]:
    """Return ``(correct, answer_present)`` for one completion."""
    payload = parse_payload(ground_truth)
    answer_kind(payload)
    span = extract_boxed(text)
    if not span:
        return 0.0, 0.0
    try:
        correct = matches(span, payload)
    except (ValueError, ArithmeticError):
        correct = False
    return (1.0 if correct else 0.0), 1.0
