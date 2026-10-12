"""Shared constants for the MedCalc Tandem RLVR run.

The training recipe matches the paper's GRPO / TRLVR setup on
Qwen3-4B-Instruct-2507. The task, prompt length, and verifier change:
MedCalc-Bench-Verified replaces DeepScaleR, and the reward is the
benchmark's calculator check on a boxed final answer.
"""

import random

DATASET_ID = "nsk7153/MedCalc-Bench-Verified"
DATA_SOURCE = "medcalc"
EXPECTED_TRAIN = 10538
EXPECTED_TEST = 1100

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BOUNDARY_COUNT = 53021

# Paper Table 3, except the prompt budget. Patient notes plus the calculator
# question do not fit in the 1,024-token math prompt cap. 6,144 leaves room
# for the longest notes; 3,000 matches the paper's response cap. The
# evaluation context has to hold both.
MAX_PROMPT = 6144
MAX_RESPONSE = 3000
CONTEXT = 10240
HOLDOUT_N = 1000
SPLIT_SEED = 42
TRAIN_STEPS = 200
SMOKE_STEPS = 3
VALIDATION_METRIC = "val-core/medcalc/acc/best@4/mean"

CATEGORIES = (
    "lab test",
    "physical",
    "date",
    "dosage",
    "risk",
    "severity",
    "diagnosis",
)
CATEGORY_TITLES = {
    "lab_test": "Lab",
    "physical": "Physical",
    "date": "Date",
    "dosage": "Dosage",
    "risk": "Risk",
    "severity": "Severity",
    "diagnosis": "Diagnosis",
}

BOXED_INSTRUCTION = (
    "Solve the clinical calculation step by step using only the patient note. "
    "Put the final answer inside \\boxed{} with no units. "
    "Use a plain number for decimal and integer results, MM/DD/YYYY for dates, "
    "and (weeks, days) when the result is a duration in weeks and days."
)


def slugify(category: str) -> str:
    text = category.strip().lower().replace(" ", "_")
    if not text or not text.replace("_", "").isalnum():
        raise ValueError(f"Unexpected category name: {category!r}")
    return text


def render_prompt(note: str, question: str) -> str:
    return (
        f"Patient note:\n{note.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        f"{BOXED_INSTRUCTION}"
    )


def holdout_mask(calculator_ids: list[int], target: int, seed: int) -> list[bool]:
    """Choose a stratified held-out set, leaving every calculator at least one train row.

    Allocation is the largest-remainder share of ``target``. The draw is
    deterministic for a fixed ``seed`` and does not depend on dict order.
    """
    if target < 0:
        raise ValueError("holdout target must be non-negative")
    total = len(calculator_ids)
    if target >= total:
        raise ValueError("holdout target must leave at least one training row")
    if target == 0:
        return [False] * total

    by_calc: dict[int, list[int]] = {}
    for index, calculator_id in enumerate(calculator_ids):
        by_calc.setdefault(int(calculator_id), []).append(index)
    rng = random.Random(seed)
    for calculator_id in sorted(by_calc):
        rng.shuffle(by_calc[calculator_id])

    capacity = {calculator_id: len(indexes) - 1 for calculator_id, indexes in by_calc.items()}
    if target > sum(capacity.values()):
        raise ValueError(
            f"holdout target {target} exceeds capacity {sum(capacity.values())}; "
            "too many calculators have only one row"
        )
    raw = {
        calculator_id: target * len(indexes) / total
        for calculator_id, indexes in by_calc.items()
    }
    alloc = {
        calculator_id: min(capacity[calculator_id], int(raw[calculator_id]))
        for calculator_id in by_calc
    }
    leftover = target - sum(alloc.values())
    order = sorted(
        by_calc,
        key=lambda calculator_id: (
            -(raw[calculator_id] - int(raw[calculator_id])),
            -len(by_calc[calculator_id]),
            calculator_id,
        ),
    )
    while leftover:
        progressed = False
        for calculator_id in order:
            if leftover <= 0:
                break
            if alloc[calculator_id] < capacity[calculator_id]:
                alloc[calculator_id] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            raise ValueError("could not allocate the held-out set")

    chosen: set[int] = set()
    for calculator_id, count in alloc.items():
        chosen.update(by_calc[calculator_id][:count])
    if len(chosen) != target:
        raise ValueError(f"held out {len(chosen)} rows, expected {target}")
    return [index in chosen for index in range(total)]


def shell_defaults() -> dict[str, str]:
    """Values train.sh reads so the shell and Python stay on one protocol."""
    return {
        "MAX_PROMPT": str(MAX_PROMPT),
        "MAX_RESPONSE": str(MAX_RESPONSE),
        "MEDCALC_CONTEXT": str(CONTEXT),
        "HOLDOUT_N": str(HOLDOUT_N),
        "SPLIT_SEED": str(SPLIT_SEED),
        "TRAIN_STEPS": str(TRAIN_STEPS),
        "SMOKE_STEPS": str(SMOKE_STEPS),
        "VALIDATION_METRIC": VALIDATION_METRIC,
        "BASE_MODEL_ID": BASE_MODEL,
        "BASE_REVISION": BASE_REVISION,
        "BOUNDARY_COUNT": str(BOUNDARY_COUNT),
        "DATASET_ID": DATASET_ID,
    }
