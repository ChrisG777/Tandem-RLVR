"""Evaluate plain-completion GSM8K with resumable traces and first/last accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import common

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.build_drift import RAW_TEMPLATE
from reward.gsm8k_drift import compute_score
from train.figure2_checkpoint import read_training_records


def main() -> None:
    """Select by validation only, or evaluate --model; write complete held-out traces.

    --select-run writes selected.json without loading a GPU. Evaluations use greedy
    decoding, the training prompt, and 256 tokens. Progress is committed per batch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-run", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.select_run:
        root = args.select_run
        records = read_training_records(root)
        candidates = []
        for step, metrics in records.items():
            path = root / "hf" / f"global_step_{step}"
            if not path.exists():
                path = root / f"global_step_{step}" / "actor" / "huggingface"
            scores = [v for k, v in metrics.items() if k.startswith("val-core/gsm8k/acc/") and "mean@1" in k]
            if scores and list(path.glob("*.safetensors")):
                candidates.append((float(scores[0]), -step, str(path)))
        if not candidates:
            raise ValueError("No validation-scored saved checkpoint; refusing final-step fallback")
        score, negative_step, path = max(candidates)
        selected = {"path": path, "step": -negative_step, "validation_accuracy": score,
                    "criterion": "highest greedy validation accuracy; earliest on ties"}
        common.save_json(root / "selected.json", selected)
        print(json.dumps(selected))
        return
    if not all((args.model, args.data, args.out)):
        parser.error("evaluation requires --model, --data, --out")
    evaluate(args.model, args.data, args.out)


def evaluate(model: str, data: Path, out: Path) -> dict:
    """Run one-GPU vLLM evaluation; reuse only identical durable progress."""
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    provenance = {"model": model, "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                  "temperature": 0, "max_tokens": 256, "seed": 17, "template": RAW_TEMPLATE}
    if out.exists():
        previous = json.loads(out.read_text())
        if previous["provenance"] != provenance:
            raise ValueError("Existing evaluation has different inputs")
        return previous
    rows = pd.read_parquet(data).to_dict("records")
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.chat_template = RAW_TEMPLATE
    prompts = [tokenizer.apply_chat_template(list(row["prompt"]), tokenize=False,
                                             add_generation_prompt=True) for row in rows]
    token_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    assert max(map(len, token_ids)) <= 1024
    problems = [{"set": "gsm8k", "idx": i, "prompt_sha256": row["extra_info"]["prompt_sha256"]}
                for i, row in enumerate(rows)]
    progress_path, progress = common.load_progress(str(out), provenance, problems)
    generations = progress["gens"]
    engine = common.build_engine(model, 0.55, max_num_batched_tokens=2048, max_model_len=1281)
    for start in range(len(generations), len(rows), common.EVAL_BATCH_SIZE):
        # The raw template already inserts BOS. Passing IDs prevents vLLM's
        # tokenizer from inserting a second BOS for base Llama/Gemma.
        outputs = engine.generate([{"prompt_token_ids": ids}
                                   for ids in token_ids[start:start + common.EVAL_BATCH_SIZE]],
                                  SamplingParams(temperature=0, max_tokens=256, seed=17))
        for i, output in enumerate(outputs, start):
            completion = output.outputs[0]
            gt = rows[i]["reward_model"]["ground_truth"]
            generations.append({**problems[i], "prompt": prompts[i], "target": gt,
                                "samples": [{"text": completion.text, "tokens": len(completion.token_ids),
                                             "finish_reason": completion.finish_reason,
                                             **compute_score("gsm8k", completion.text, gt)}]})
        common.save_json(progress_path, progress)
    samples = [g["samples"][0] for g in generations]
    metrics = {key: sum(s[key] for s in samples) / len(samples)
               for key in ("acc", "first_acc", "tokens", "answer_count", "format_ok")}
    metrics["truncated"] = sum(s["finish_reason"] == "length" for s in samples) / len(samples)
    result = {"provenance": provenance, "metrics": metrics, "generations": generations}
    common.save_json(out, result)
    print(json.dumps(metrics))
    return result


if __name__ == "__main__":
    main()
