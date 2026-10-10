"""Prepare pinned GSM8K splits and optional base weights for natural-drift pilots."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

DATA_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MODELS = {
    "llama": ("meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"),
    "gemma": ("google/gemma-3-1b-pt", "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"),
    "qwen": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
}
# Base-model continuation, with one BOS and no instruction/chat-role tokens.
RAW_TEMPLATE = "{{ bos_token or '' }}{% for message in messages %}{{ message['content'] }}{% endfor %}"


def main() -> None:
    """Write disjoint Parquet splits/manifest; optionally download pinned weights.

    --out is an immutable campaign data directory. Repeated preparation checks
    existing file hashes. Model download requires the user's gated-model access.
    """
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if not manifest_path.exists():
        ds = load_dataset("openai/gsm8k", "main", revision=DATA_REVISION)
        assert len(ds["train"]) == 7473 and len(ds["test"]) == 1319
        # Fixed ordinary worked examples, excluded from training and validation.
        examples = [ds["train"][i] for i in (0, 1)]
        prefix = "Solve each problem, showing your reasoning. Finish with a line containing #### followed by the numeric answer.\n\n"
        for ex in examples:
            answer = re.sub(r"<<.*?>>", "", ex["answer"])
            prefix += f"Question: {ex['question']}\nAnswer: {answer}\n\n"
        indices = list(range(2, len(ds["train"])))
        random.Random(42).shuffle(indices)
        split_ids = {"validation": indices[:128], "train": indices[128:],
                     "test": list(range(len(ds["test"])))}
        manifest = {"dataset": "openai/gsm8k", "revision": DATA_REVISION,
                    "split_seed": 42, "demo_indices": [0, 1], "sha256": {},
                    "raw_template": RAW_TEMPLATE, "split_indices": split_ids,
                    "prompt_deviation": "Two fixed GSM8K worked examples instead of paper's synthetic examples."}
        for split, ids in split_ids.items():
            source = ds["test" if split == "test" else "train"]
            rows = []
            for idx in ids:
                ex = source[idx]
                prompt = prefix + f"Question: {ex['question']}\nAnswer:"
                target = ex["answer"].split("####")[-1].strip().replace(",", "")
                rows.append({"data_source": "gsm8k", "ability": "math",
                             "prompt": [{"role": "user", "content": prompt}],
                             "reward_model": {"style": "rule", "ground_truth": target},
                             "extra_info": {"index": idx, "split": split,
                                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}})
            path = args.out / f"{split}.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        temporary = manifest_path.with_suffix(".partial")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["revision"] == DATA_REVISION and manifest["raw_template"] == RAW_TEMPLATE
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((args.out / name).read_bytes()).hexdigest() == digest, name
    if args.model:
        repo_id, revision = MODELS[args.model]
        path = snapshot_download(repo_id, revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.chat_template = RAW_TEMPLATE
        lengths = [len(tokenizer.apply_chat_template(row, tokenize=True, add_generation_prompt=True))
                   for row in pd.read_parquet(args.out / "train.parquet")["prompt"]]
        assert max(lengths) <= 1024, f"Prompt would be filtered: {max(lengths)}"
        info = {"repo_id": repo_id, "revision": revision, "path": path,
                "max_train_prompt_tokens": max(lengths)}
        target = args.out / f"{args.model}.json"
        tmp = target.with_suffix(".partial")
        tmp.write_text(json.dumps(info, indent=2) + "\n")
        tmp.replace(target)
        print(json.dumps(info))
    print(f"Verified GSM8K data: {args.out}")


if __name__ == "__main__":
    main()
