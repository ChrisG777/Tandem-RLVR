"""Strict final-answer grading for repeated reasoning pilots."""
import re

TASKS = {"manipulate_matrix", "string_manipulation", "ruletaker_shared", "rearc_objects"}
FINAL = re.compile(r"<answer>(.*?)</answer>\s*\Z", re.DOTALL)


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict | None = None) -> dict[str, float]:
    """Return correctness/format diagnostics; malformed answers receive zero.

    Accept exactly one final answer block. Matrix whitespace is insignificant,
    but row boundaries and entries are significant. Strings are case-sensitive.
    RuleTaker earns the fraction of four questions correct; acc still means
    the entire answer is correct. Other tasks use exact-match reward.
    No substring, style, or shorthand rewards are used.
    Unknown tasks raise rather than silently train against an incorrect verifier.
    """
    if data_source not in TASKS:
        raise ValueError(f"Unknown shorthand task: {data_source}")
    answer = extract_answer(solution_str)
    present = answer is not None
    if data_source == "ruletaker_shared":
        target = ground_truth.split()
        if len(target) != 4 or any(x not in {"true", "false"} for x in target):
            raise ValueError("Invalid RuleTaker ground truth")
        values = answer.split() if present else []
        valid = len(values) == 4 and all(x in {"true", "false"} for x in values)
        score = sum(x == y for x, y in zip(values, target)) / 4 if valid else 0.0
        return {"score": score, "acc": float(valid and values == target),
                "query_accuracy": score, "answer_present": float(present)}
    correct = present and normalize(answer, data_source) == normalize(ground_truth, data_source)
    return {"score": float(correct), "acc": float(correct), "answer_present": float(present)}


def extract_answer(text: str) -> str | None:
    """Extract the sole complete answer block at the end, or return None."""
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    match = FINAL.search(text)
    return match.group(1).strip() if match else None


def normalize(answer: str, task: str) -> str:
    """Normalize spaces between matrix entries without flattening rows."""
    if task in {"manipulate_matrix", "rearc_objects"}:
        return "\n".join(" ".join(line.split()) for line in answer.strip().splitlines())
    return answer.strip()
