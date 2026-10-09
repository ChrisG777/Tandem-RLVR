"""Evaluate base/trained seniors alone at matched budgets, retaining raw traces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

import common
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reward"))
from shorthand_reward import compute_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gpu-util", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=common.MAX_MODEL_LEN)
    args = parser.parse_args()
    # One engine handles all splits and budgets; outputs are independent files.
    os.environ.pop("VLLM_TANDEM_CONFIG", None)
    os.environ.pop("VLLM_TANDEM_ALL_GPUS", None)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = common.build_engine(args.model, args.gpu_util, max_num_batched_tokens=4096,
                                 max_model_len=args.max_model_len)
    for path in args.data:
        for budget in args.budgets:
            out = args.out_dir / f"{path.parent.name}-{path.stem}-b{budget}.json"
            result = evaluate(args.model, path, out, budget, args.seed,
                              engine=engine, tokenizer=tokenizer, n=args.n, max_model_len=args.max_model_len)
            print(json.dumps({"output": str(out), **result["metrics"]}), flush=True)


def evaluate(model: str, data_path: Path, out_path: Path, max_tokens: int,
             seed: int = 17, *, engine=None, tokenizer=None, n: int = 4,
             max_model_len: int = common.MAX_MODEL_LEN) -> dict:
    """Generate solo GPU rollouts and atomically write scores, lengths, and full traces.

    Uses temperature 0.6/top-p 1 for both base and trained policies. Existing
    complete files may be reused only when their inputs match exactly.
    """
    from vllm import SamplingParams
    provenance = {"model": model, "data": str(data_path),
                  "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  "max_tokens": max_tokens, "seed": seed, "n": n, "temperature": 0.6,
                  "top_p": 1.0, "phase": "solo"}
    # Preserve compatibility with existing 4k-context pilot artifacts.
    if max_model_len != common.MAX_MODEL_LEN:
        provenance["max_model_len"] = max_model_len
    if out_path.exists():
        previous = json.loads(out_path.read_text())
        if previous["provenance"] != provenance:
            raise ValueError(f"Refusing to overwrite different evaluation: {out_path}")
        return previous
    os.environ.pop("VLLM_TANDEM_CONFIG", None)
    os.environ.pop("VLLM_TANDEM_ALL_GPUS", None)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
    if engine is None:
        engine = common.build_engine(model, 0.75, max_num_batched_tokens=4096,
                                     max_model_len=max_model_len)
    rows = pd.read_parquet(data_path).to_dict("records")
    prompts = [common.chat_prefix(tokenizer, row["prompt"][0]["content"]) for row in rows]
    lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    if not rows or max(lengths) + max_tokens > max_model_len:
        raise ValueError("Empty dataset or evaluation exceeds context length")
    problems = [{"set": data_path.parent.name, "idx": i,
                 "prompt_sha256": row["extra_info"]["prompt_sha256"]}
                for i, row in enumerate(rows)]
    progress_path, progress = common.load_progress(str(out_path), provenance, problems)
    generations = progress["gens"]
    for start in range(len(generations), len(rows), common.EVAL_BATCH_SIZE):
        stop = start + common.EVAL_BATCH_SIZE
        outputs = engine.generate(prompts[start:stop], SamplingParams(n=n, temperature=0.6, top_p=1.0,
                                  top_k=-1, max_tokens=max_tokens, seed=seed))
        batch = []
        for i, output in enumerate(outputs, start):
            row, prompt_tokens = rows[i], lengths[i]
            samples = []
            for completion in output.outputs:
                score = compute_score(row["data_source"], completion.text, row["reward_model"]["ground_truth"])
                samples.append({"text": completion.text, "tokens": len(completion.token_ids),
                                "finish_reason": completion.finish_reason, **score})
            batch.append({**problems[i], "prompt": row["prompt"][0]["content"],
                          "target": row["reward_model"]["ground_truth"],
                          "prompt_tokens": prompt_tokens, "samples": samples})
        if len(batch) != len(rows[start:stop]):
            raise ValueError("Incomplete generation batch")
        generations.extend(batch)
        common.save_json(progress_path, progress)
    samples = [s for g in generations for s in g["samples"]]
    total = len(samples)
    metrics = {"accuracy": sum(s["acc"] for s in samples) / total,
               "mean_reward": sum(s["score"] for s in samples) / total,
               "pass_at_n": sum(any(s["acc"] for s in g["samples"]) for g in generations) / len(rows),
               "answer_present": sum(s["answer_present"] for s in samples) / total,
               "mean_tokens": sum(s["tokens"] for s in samples) / total,
               "truncated": sum(s["finish_reason"] == "length" for s in samples) / total,
               "max_prompt_tokens": max(lengths), "problems": len(rows), "samples": total}
    if "query_accuracy" in samples[0]:
        metrics["query_accuracy"] = sum(s["query_accuracy"] for s in samples) / total
    families = sorted({row["extra_info"].get("family", "all") for row in rows})
    metrics["by_family"] = {}
    for family in families:
        subset = [s for row, generation in zip(rows, generations)
                  if row["extra_info"].get("family", "all") == family for s in generation["samples"]]
        metrics["by_family"][family] = {"accuracy": sum(s["acc"] for s in subset) / len(subset),
                                         "samples": len(subset)}
    result = {"provenance": provenance, "metrics": metrics, "generations": generations}
    common.save_json(out_path, result)
    return result


if __name__ == "__main__":
    main()
