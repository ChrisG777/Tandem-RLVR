import json
import hashlib
import math
import os
import sys
from pathlib import Path

import pandas as pd

import config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"

sys.path.insert(0, str(ROOT / "reward"))

AIME_SETS = ("aime24", "aime25", "aime26")
DEFAULT_SETS = (*AIME_SETS, "amc23_25", "minerva", "olympiad")
ALL_SETS = ("aime24", "aime25", "aime26", "amc23_25", "math500", "minerva", "olympiad")

KS = (1, 2, 4, 8, 16, 32)
EVAL_BATCH_SIZE = 16

MAX_MODEL_LEN = 4096
MAX_PROMPT = None
CONTEXT_RESERVE = 8


def load_progress(out_path: str, identity: dict, problems: list[dict], *,
                  batch_size: int = EVAL_BATCH_SIZE) -> tuple[Path, dict]:
    """Load compatible completed evaluation batches; reject changed inputs.

    Progress lives beside the final JSON and is committed atomically after each
    batch. Only complete batches are reused; interruption repeats at most one.
    """
    path = Path(str(out_path) + ".progress.json")
    signature = hashlib.sha256(json.dumps(
        {"identity": identity, "problems": problems, "batch_size": batch_size},
        sort_keys=True).encode()).hexdigest()
    if path.exists():
        progress = json.loads(path.read_text())
        if progress["signature"] != signature:
            raise ValueError(f"Evaluation inputs differ from saved progress: {path}")
        gens = progress["gens"]
        if len(gens) > len(problems) or any(
            (g["set"], g["idx"]) != (p["set"], p["idx"])
            for g, p in zip(gens, problems)
        ):
            raise ValueError(f"Invalid evaluation progress: {path}")
        return path, progress
    return path, {"signature": signature, "gens": [], "unfinished_chains": 0}


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
                    "data_source": r.get("data_source", "math"),
                    **({"task_id": r["extra_info"]["task_id"]}
                       if r.get("data_source") == "apps" else {}),
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


def grade(text, gt, data_source="math"):
    if data_source == "apps":
        return float(grade_details(text, gt, data_source)["acc"])
    from math_boxed_reward import compute_score
    return float(compute_score("math", text, gt)["acc"])


def grade_details(text, gt, data_source="math"):
    if data_source == "apps":
        sys.path.insert(0, str(ROOT))
        from experiments.coding.reward import compute_score as coding_score
        return coding_score(data_source, text, gt)
    return {"acc": grade(text, gt)}


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


def build_engine(model, gpu_util, device=None, max_num_batched_tokens=None,
                 max_model_len=None):
    from vllm import LLM

    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    try:
        kwargs = dict(
            model=model,
            gpu_memory_utilization=gpu_util,
            enforce_eager=True,
            max_model_len=max_model_len or MAX_MODEL_LEN,
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


def add_data_arguments(parser):
    """Optional dataset/context overrides; math defaults stay unchanged."""
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-prompt", type=int)
    parser.add_argument("--max-response", type=int, default=3000)


def configure(args):
    global DATA, MAX_MODEL_LEN, MAX_PROMPT
    DATA = args.data_root or ROOT / "data/eval"
    MAX_MODEL_LEN, MAX_PROMPT = args.context, args.max_prompt
    config.MAX_TOKENS = getattr(args, "max_response", 3000)
    if config.MAX_TOKENS < 1:
        raise ValueError("Response budget must be positive")
    if MAX_MODEL_LEN < config.MAX_TOKENS + CONTEXT_RESERVE:
        raise ValueError("Context smaller than response budget")


def evaluation_identity(problems):
    metadata = {"data_root": str(DATA.resolve()), "context": MAX_MODEL_LEN, "max_prompt": MAX_PROMPT}
    if any(p.get("data_source") == "apps" for p in problems):
        sys.path.insert(0, str(ROOT))
        from experiments.coding.reward import grader_identity
        metadata["grader"] = grader_identity()
        metadata["dataset_sha256"] = hashlib.sha256((DATA.parent / "manifest.json").read_bytes()).hexdigest()
    return metadata


def overlength(tokenizer, prompt):
    if MAX_PROMPT is None:
        return False
    length = len(tokenizer.encode(prompt))
    return length > (MAX_PROMPT if MAX_PROMPT is not None else MAX_MODEL_LEN - CONTEXT_RESERVE - 1)


def coverage(gens):
    """Count length exclusions explicitly while retaining them in score denominators."""
    excluded = sum(bool(g.get("overlength")) or any(d.get("overlength") for d in g.get("diagnostics", [])) for g in gens)
    return {"total_problems": len(gens), "attempted_problems": len(gens)-excluded,
            "overlength_problems": excluded}
