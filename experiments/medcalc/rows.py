"""Turn one MedCalc-Bench row into the parquet record verl already trains on."""

import json

from experiments.medcalc.grade import validate_gold
from experiments.medcalc.protocol import DATA_SOURCE, render_prompt, slugify


def _text(row: dict, key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def make_record(row: dict, index: int) -> dict:
    note = _text(row, "Patient Note")
    question = _text(row, "Question")
    if not note or not question:
        raise ValueError(f"row {index} is missing the patient note or the question")
    payload = {
        "answer": _text(row, "Ground Truth Answer"),
        "lower": _text(row, "Lower Limit"),
        "upper": _text(row, "Upper Limit"),
        "calculator_id": int(row["Calculator ID"]),
        "output_type": _text(row, "Output Type"),
    }
    validate_gold(payload)
    category = _text(row, "Category")
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": render_prompt(note, question)}],
        "ability": "medcalc",
        "reward_model": {
            "ground_truth": json.dumps(payload, separators=(",", ":")),
            "style": "rule",
        },
        "extra_info": {
            "index": index,
            "task_id": int(row["Row Number"]),
            "calculator_id": payload["calculator_id"],
            "calculator_name": _text(row, "Calculator Name"),
            "category": category,
            "category_slug": slugify(category),
            "output_type": payload["output_type"],
            "note_id": _text(row, "Note ID"),
        },
    }
