"""Senior/junior token agreement on saved word-handoff responses, by answer region.

Scores every response token under both the senior and the junior (teacher
forcing; no generation). A junior-written token is "surprising" when the senior
gives it probability below --surprise. Reports per-response exposure (code or
\\boxed{} answer vs reasoning), top-choice disagreement rates, and whether
surprising junior tokens predict failure within a problem.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
import common
from experiments.recovery.authorship import answer_span, token_bytes, within_problem_effects

REGIONS = ("answer", "reasoning")
CHUNK = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="word-handoff result JSON")
    parser.add_argument("--domain", choices=("apps", "math"), required=True)
    parser.add_argument("--senior", required=True)
    parser.add_argument("--junior", required=True)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--problems", type=int, default=0, help="random subset size; 0 = all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--surprise", type=float, default=0.01, help="senior-probability threshold")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    common.add_data_arguments(parser)
    args = parser.parse_args()
    common.configure(args)
    rows = score(args)
    report = summarize(rows, args)
    common.save_json(args.out, {**report, "samples": rows})
    print(json.dumps(report, indent=1))


def score(args) -> list[dict]:
    """GPU: per-sample region statistics from both models' teacher-forced log-probs."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    result = json.loads(args.result.read_text())
    problems = {p["idx"]: p for p in common.load_problems(common.parse_sets(args.benchmarks))}
    gens = result["gens"]
    if any(problems[g["idx"]]["set"] != g["set"] for g in gens):
        raise ValueError("Result does not align with --benchmarks problem order")
    if args.problems:
        rng = np.random.default_rng(args.seed)
        gens = [gens[i] for i in sorted(rng.choice(len(gens), size=min(args.problems, len(gens)), replace=False))]
    tok = AutoTokenizer.from_pretrained(args.senior)
    pieces = token_bytes(tok)
    work = []
    for g in gens:
        prompt = tok.encode(common.chat_prefix(tok, problems[g["idx"]]["content"]), add_special_tokens=False)
        for sample, correct in zip(g["samples"], g["correct"]):
            work.append((g["idx"], float(correct), prompt, sample["token_ids"], sample["model_mask"]))
    # One model at a time keeps memory to one 4B model (~15 GiB), so 24 GiB cards suffice.
    # Prompt log-probs materialize full-vocabulary logits per prefill chunk; keep chunks small.
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    scored = {}
    for role, path in (("senior", args.senior), ("junior", args.junior)):
        llm = common.build_engine(path, 0.85, max_num_batched_tokens=4096, max_model_len=common.MAX_MODEL_LEN)
        scored[role] = []
        for start in range(0, len(work), CHUNK):
            chunk = work[start:start + CHUNK]
            prompts = [TokensPrompt(prompt_token_ids=p + r) for _, _, p, r, _ in chunk]
            scored[role] += [extract(o, len(p), r) for o, (_, _, p, r, _)
                             in zip(llm.generate(prompts, params, use_tqdm=False), chunk)]
            print(f"{role}: scored {min(start + CHUNK, len(work))}/{len(work)} responses", flush=True)
        llm.llm_engine.engine_core.shutdown()
        del llm
    rows = []
    for k, (idx, correct, _, response, mask) in enumerate(work):
        features = region_features(response, mask, scored["senior"][k], scored["junior"][k],
                                   pieces, args.domain, args.surprise)
        if features is not None:
            rows.append({"problem": idx, "correct": correct, **features})
    return rows


def extract(output, prompt_len: int, response: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Log-prob of each realized response token and the model's top-choice token id."""
    logprobs, top = np.empty(len(response)), np.empty(len(response), dtype=np.int64)
    for i, token in enumerate(response):
        entry = output.prompt_logprobs[prompt_len + i]
        logprobs[i] = entry[token].logprob
        top[i] = next(t for t, v in entry.items() if v.rank == 1)
    return logprobs, top


def region_features(response, mask, senior, junior, pieces, domain, surprise) -> dict | None:
    """Counts per region: tokens, junior tokens, surprising junior tokens, top-choice disagreements."""
    lengths = np.array([len(pieces.get(t, b"")) for t in response])
    span = answer_span(b"".join(pieces.get(t, b"") for t in response), domain)
    if span is None:
        return None
    ends = np.cumsum(lengths)
    starts = ends - lengths
    regions = {"answer": (ends > span[0]) & (starts < span[1]) & (lengths > 0),
               "reasoning": (ends <= span[0]) & (lengths > 0)}
    is_junior = np.asarray(mask) == 0
    s_lp, s_top = senior
    j_lp, j_top = junior
    out = {}
    for name, region in regions.items():
        jr, sr = region & is_junior, region & ~is_junior
        out |= {f"{name}_tokens": int(region.sum()), f"{name}_junior_tokens": int(jr.sum()),
                f"{name}_surprising_junior": int((jr & (s_lp < np.log(surprise))).sum()),
                f"{name}_surprising_senior": int((sr & (j_lp < np.log(surprise))).sum()),
                f"{name}_top_disagree": int((region & (s_top != j_top)).sum()),
                f"{name}_junior_senior_logprob": float(s_lp[jr].sum())}
    return out


def summarize(rows: list[dict], args) -> dict:
    """Per-response exposure, rates, and within-problem effects of surprising junior tokens."""
    def mean(key):
        return float(np.mean([r[key] for r in rows]))
    regions = {}
    for name in REGIONS:
        tokens, junior = sum(r[f"{name}_tokens"] for r in rows), sum(r[f"{name}_junior_tokens"] for r in rows)
        regions[name] = {
            "tokens_per_response": mean(f"{name}_tokens"),
            "junior_tokens_per_response": mean(f"{name}_junior_tokens"),
            "surprising_junior_per_response": mean(f"{name}_surprising_junior"),
            "surprising_senior_per_response": mean(f"{name}_surprising_senior"),
            "surprising_junior_rate": sum(r[f"{name}_surprising_junior"] for r in rows) / max(junior, 1),
            "top_choice_disagreement_rate": sum(r[f"{name}_top_disagree"] for r in rows) / max(tokens, 1),
            "responses_with_any_surprising_junior": float(np.mean([r[f"{name}_surprising_junior"] > 0 for r in rows]))}
    names = ("answer_surprising_junior", "reasoning_surprising_junior", "answer_junior_tokens", "reasoning_junior_tokens")
    X = np.array([[r[n] for n in names] for r in rows], dtype=float)
    y = np.array([r["correct"] for r in rows])
    groups = np.array([r["problem"] for r in rows])
    effects = within_problem_effects(X, y, groups, args.bootstrap, args.seed, names=names, per_sd=False)
    buckets = {}
    for label, lo, hi in (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4+", 4, 10**9)):
        keep = [(lo <= r["answer_surprising_junior"] <= hi) for r in rows]
        if any(keep):
            buckets[label] = {"responses": int(sum(keep)), "success": float(np.mean(y[np.array(keep)]))}
    return {"domain": args.domain, "senior": args.senior, "junior": args.junior,
            "surprise_threshold": args.surprise, "responses": len(rows),
            "problems": len(set(groups.tolist())), "success_rate": float(y.mean()),
            "regions": regions, "effects_pp_per_token": effects,
            "success_by_surprising_junior_answer_tokens": buckets}


if __name__ == "__main__":
    main()
