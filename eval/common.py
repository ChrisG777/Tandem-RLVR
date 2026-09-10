import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

import config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"

sys.path.insert(0, str(ROOT / "reward"))
from math_boxed_reward import compute_score

AIME_SETS = ("aime24", "aime25", "aime26")
DEFAULT_SETS = (*AIME_SETS, "amc23_25", "minerva", "olympiad")
ALL_SETS = ("aime24", "aime25", "aime26", "amc23_25", "math500", "minerva", "olympiad")

KS = (1, 2, 4, 8)

MAX_MODEL_LEN = 4096
CONTEXT_RESERVE = 8


def parse_sets(spec):
    names = tuple(s.strip() for s in spec.split(",") if s.strip())
    unknown = [n for n in names if not (DATA / n / "test.parquet").exists()]
    if unknown:
        raise SystemExit(
            f"unknown benchmark(s) {', '.join(unknown)}; available: {', '.join(ALL_SETS)}"
        )
    return names


def load_problems(sets=DEFAULT_SETS, limit=0):
    rows = []
    for name in sets:
        df = pd.read_parquet(DATA / name / "test.parquet")
        for pos, (_, r) in enumerate(df.iterrows()):
            rows.append(
                {
                    "set": name,
                    "row": pos,
                    "idx": len(rows),
                    "content": r["prompt"][0]["content"],
                    "gt": str(r["reward_model"]["ground_truth"]),
                }
            )
    return rows[:limit] if limit else rows


def problems_from_result(result):
    return load_problems(tuple(result["benchmarks"]), result.get("limit", 0))


def chat_prefix(tokenizer, content):
    messages = [{"role": "user", "content": content}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, **kwargs)


def response_budget(prompt_len):
    return min(config.MAX_TOKENS, MAX_MODEL_LEN - prompt_len - CONTEXT_RESERVE)


def grade(text, gt):
    return float(compute_score("math", text, gt)["acc"])


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def agg_pass_at_k(counts, n, ks=KS):
    return {
        f"pass@{k}": sum(pass_at_k(n, c, k) for c in counts) / max(len(counts), 1)
        for k in ks
        if k <= n
    }


def group_by_benchmark(values_by_set):
    grouped = {}
    for name, values in values_by_set.items():
        benchmark = "aime24_26" if name in AIME_SETS else name
        grouped.setdefault(benchmark, []).extend(values)
    return grouped


def metrics_by_set(problems, counts, n, ks=KS):
    grouped = {}
    for p, c in zip(problems, counts):
        grouped.setdefault(p["set"], []).append(c)
    by_set = {name: agg_pass_at_k(cs, n, ks) for name, cs in grouped.items()}
    by_benchmark = {
        name: agg_pass_at_k(cs, n, ks)
        for name, cs in group_by_benchmark(grouped).items()
    }
    out = dict(by_set)
    out["by_benchmark"] = by_benchmark
    out["macro"] = {
        key: sum(v[key] for v in by_benchmark.values()) / len(by_benchmark)
        for key in next(iter(by_benchmark.values()))
    }
    return out


def build_engine(model, gpu_util, device=None, max_num_batched_tokens=None):
    from vllm import LLM

    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    try:
        kwargs = dict(
            model=model,
            gpu_memory_utilization=gpu_util,
            enforce_eager=True,
            max_model_len=MAX_MODEL_LEN,
            enable_prefix_caching=True,
        )
        if max_num_batched_tokens:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        return LLM(**kwargs)
    finally:
        if device is not None:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved


def visible_devices():
    spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not spec:
        return []
    return [d for d in spec.split(",") if d.strip()]


def save_json(path, obj):
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)
