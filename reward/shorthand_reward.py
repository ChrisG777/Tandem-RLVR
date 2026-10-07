"""Strict binary final-answer grading for repeated-operation pilot tasks."""
import re

TASKS = {"manipulate_matrix", "string_manipulation"}
FINAL = re.compile(r"<answer>(.*?)</answer>\s*\Z", re.DOTALL)


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict | None = None) -> dict[str, float]:
    """Return correctness/format diagnostics; malformed answers receive zero.

    Accept exactly one final answer block. Matrix whitespace is insignificant,
    but row boundaries and entries are significant. Strings are case-sensitive.
    No partial-credit, substring, style, or shorthand rewards are used.
    Unknown tasks raise rather than silently train against an incorrect verifier.
    """
    if data_source not in TASKS:
        raise ValueError(f"Unknown shorthand task: {data_source}")
    answer = extract_answer(solution_str)
    present = answer is not None
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
    if task == "manipulate_matrix":
        return "\n".join(" ".join(line.split()) for line in answer.strip().splitlines())
    return answer.strip()
