"""Test 2: after one forced plausible wrong token, can the model still finish correctly?

From correct base-model solo responses, cut once inside the answer region (APPS:
final ```python block; math: last \\boxed{}) and once in the reasoning before it.
At each cut the original next token is replaced by a different top-20 token
(sampled by renormalized model probability); the model then continues freely.
With --junior, cuts are restricted to senior/junior disagreements and the junior's top
choice is forced instead (a realistic handoff intrusion). A matched control
continues from the same prefix without the forced token.
Recovery = perturbed success / control success, per region and domain.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
import common
import config
from experiments.recovery.authorship import answer_span, token_bytes

TOP_K = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="GPU: pick cuts, force tokens, sample continuations")
    gen.add_argument("--model", required=True)
    gen.add_argument("--domain", choices=("apps", "math"), required=True)
    gen.add_argument("--source", type=Path, nargs="+", required=True,
                     help="solo result JSON(s) or shard progress files with texts/correct")
    gen.add_argument("--benchmarks", required=True)
    gen.add_argument("--problems", type=int, default=400)
    gen.add_argument("--n", type=int, default=4, help="continuations per condition")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--junior", help="Force the junior's top choice at a senior/junior disagreement "
                     "instead of a sampled alternative from --model's own top-20")
    gen.add_argument("--out", type=Path, required=True)
    common.add_data_arguments(gen)
    grade = sub.add_parser("grade", help="CPU: grade continuations and summarize recovery")
    grade.add_argument("--input", type=Path, required=True)
    grade.add_argument("--out", type=Path, required=True)
    grade.add_argument("--bootstrap", type=int, default=2000)
    common.add_data_arguments(grade)
    args = parser.parse_args()
    common.configure(args)
    if args.command == "generate":
        generate(args)
    else:
        grade_and_summarize(args.input, args.out, args.bootstrap)


def generate(args) -> None:
    """Write cut/forced-token records and raw continuation texts (no grading)."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sets = common.parse_sets(args.benchmarks)
    problems = {p["idx"]: p for p in common.load_problems(sets)}
    gens = [g for path in args.source for g in json.loads(path.read_text())["gens"]]
    if any(problems[g["idx"]]["set"] != g["set"] for g in gens):
        raise ValueError("Source generations do not align with --benchmarks problem order")
    candidates = sorted({g["idx"]: g for g in gens if max(g["correct"]) == 1}.values(), key=lambda g: g["idx"])
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(candidates), size=min(args.problems, len(candidates)), replace=False)
    tok = AutoTokenizer.from_pretrained(args.model)
    pieces = token_bytes(tok)
    items = []
    for c in sorted(chosen):
        g = candidates[c]
        p = problems[g["idx"]]
        text = g["texts"][int(np.argmax(g["correct"]))]
        response = tok.encode(text, add_special_tokens=False)
        lengths = np.array([len(pieces.get(t, b"")) for t in response])
        span = answer_span(b"".join(pieces.get(t, b"") for t in response), args.domain)
        if span is None:
            continue
        ends = np.cumsum(lengths)
        starts = ends - lengths
        regions = {"answer": np.flatnonzero((ends > span[0]) & (starts < span[1]) & (lengths > 0)),
                   "reasoning": np.flatnonzero((ends <= span[0]) & (np.arange(len(response)) >= 4) & (lengths > 0))}
        prompt = tok.encode(common.chat_prefix(tok, p["content"]), add_special_tokens=False)
        items.append({"set": p["set"], "idx": p["idx"], "gt": p["gt"], "prompt": prompt,
                      "response": response, "regions": regions})
    if args.junior:
        junior_choices(items, args.junior)
    cuts = []
    for item in items:
        for region, positions in item["regions"].items():
            if args.junior:  # only where the junior would have written something else
                positions = positions[item["junior_top"][positions] != np.asarray(item["response"])[positions]]
            if len(positions):
                t = int(rng.choice(positions))
                cut = {k: item[k] for k in ("set", "idx", "gt", "prompt")}
                cut.update(region=region, cut=t, response_tokens=len(item["response"]),
                           prefix=item["response"][:t], original=item["response"][t])
                if args.junior:
                    cut["alt"] = int(item["junior_top"][t])
                cuts.append(cut)
    items = cuts
    llm = common.build_engine(args.model, 0.85)
    probe = llm.generate([TokensPrompt(prompt_token_ids=i["prompt"] + i["prefix"]) for i in items],
                         SamplingParams(temperature=0, max_tokens=1, logprobs=TOP_K))
    kept = []
    for item, out in zip(items, probe):
        top = out.outputs[0].logprobs[0]
        alternatives = {t: v.logprob for t, v in top.items() if t != item["original"]}
        if "alt" in item:  # junior's choice; below the senior's top-20 its logprob is bounded above
            alt = item["alt"]
            below = alt not in top
            alt_logprob = min(v.logprob for v in top.values()) if below else top[alt].logprob
        elif alternatives:
            ids = list(alternatives)
            probs = np.exp(np.array([alternatives[t] for t in ids]))
            alt, below = int(rng.choice(ids, p=probs / probs.sum())), False
            alt_logprob = alternatives[alt]
        else:
            continue
        ranked = sorted(top, key=lambda t: -top[t].logprob)
        item.update(alt=alt, alt_logprob=alt_logprob, alt_below_top_k=below,
                    alt_rank=None if below else ranked.index(alt) + 1,
                    original_logprob=top[item["original"]].logprob if item["original"] in top else None,
                    original_text=tok.decode([item["original"]]), alt_text=tok.decode([alt]))
        kept.append(item)
    budget = config.MAX_TOKENS
    prompts, params, slots = [], [], []
    for k, item in enumerate(kept):
        for condition, extra in (("control", []), ("perturbed", [item["alt"]])):
            used = len(item["prefix"]) + len(extra)
            prompts.append(TokensPrompt(prompt_token_ids=item["prompt"] + item["prefix"] + extra))
            params.append(config.sampling_params(n=args.n, seed=args.seed + 7919 * k,
                                                 max_tokens=max(1, budget - used)))
            slots.append((item, condition, item["prefix"] + extra))
    for (item, condition, head), out in zip(slots, llm.generate(prompts, params)):
        item[f"{condition}_texts"] = [tok.decode(head + list(o.token_ids), skip_special_tokens=True)
                                      for o in out.outputs]
        item[f"{condition}_finish"] = [o.finish_reason for o in out.outputs]
    for item in kept:
        del item["prompt"]
    common.save_json(args.out, {"model": args.model, "junior": args.junior,
                                "domain": args.domain, "benchmarks": list(sets),
                                "n": args.n, "seed": args.seed, "top_k": TOP_K, "sampling": config.record(),
                                "context": common.MAX_MODEL_LEN, "items": kept})
    print(f"wrote {len(kept)} cuts from {len(chosen)} problems to {args.out}", flush=True)


def junior_choices(items: list[dict], junior: str) -> None:
    """Add the junior's top-choice token at every response position (teacher forcing)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    # Prompt log-probs need headroom outside vLLM's reservation (see agreement.py).
    llm = common.build_engine(junior, 0.55, max_num_batched_tokens=2048)
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    for start in range(0, len(items), 128):
        chunk = items[start:start + 128]
        outs = llm.generate([TokensPrompt(prompt_token_ids=i["prompt"] + i["response"]) for i in chunk],
                            params, use_tqdm=False)
        for item, out in zip(chunk, outs):
            n = len(item["prompt"])
            item["junior_top"] = np.array([next(t for t, v in out.prompt_logprobs[n + k].items() if v.rank == 1)
                                           for k in range(len(item["response"]))])
    llm.llm_engine.engine_core.shutdown()
    del llm


def grade_and_summarize(source: Path, out: Path, bootstrap: int) -> dict:
    """Grade both conditions with the protocol graders; report recovery with problem-bootstrap SEs."""
    data = json.loads(source.read_text())
    items = data["items"]
    source_kind = "apps" if data["domain"] == "apps" else "math"
    jobs = [(t, i["gt"], source_kind) for i in items for c in ("control", "perturbed") for t in i[f"{c}_texts"]]
    grades = iter(d["acc"] for d in common.grade_many(jobs)) if source_kind == "apps" else \
        iter(common.grade(t, gt) for t, gt, _ in jobs)
    for item in items:
        for condition in ("control", "perturbed"):
            item[f"{condition}_correct"] = [float(next(grades)) for _ in item[f"{condition}_texts"]]
    summary = {"model": data["model"], "junior": data.get("junior"), "domain": data["domain"],
               "n": data["n"], "regions": {}}
    rng = np.random.default_rng(0)
    for region in ("reasoning", "answer"):
        rows = [i for i in items if i["region"] == region]
        groups = {}
        for i in rows:
            groups.setdefault(i["idx"], []).append(i)
        def rates(selected):
            control = np.mean([np.mean(i["control_correct"]) for i in selected])
            perturbed = np.mean([np.mean(i["perturbed_correct"]) for i in selected])
            return control, perturbed
        control, perturbed = rates(rows)
        keys = list(groups)
        draws = []
        for _ in range(bootstrap):
            sample = [i for k in rng.choice(keys, size=len(keys)) for i in groups[k]]
            c, p = rates(sample)
            draws.append((c, p, p / c if c else np.nan))
        draws = np.array(draws)
        by_rank = {}
        for label, keep in (("rank2", lambda i: i["alt_rank"] == 2), ("rank3+", lambda i: (i["alt_rank"] or 99) > 2)):
            chosen = [i for i in rows if keep(i)]
            if chosen:
                c, p = rates(chosen)
                by_rank[label] = {"cuts": len(chosen), "control": c, "perturbed": p}
        by_prob = {}
        for label, lo, hi in (("<1%", 0, 0.01), ("1-10%", 0.01, 0.1), (">=10%", 0.1, 1.01)):
            # A token below the senior's top-20 has only an upper bound; count it only when that bound is < 1%.
            chosen = [i for i in rows if lo <= np.exp(i["alt_logprob"]) < hi
                      and (not i.get("alt_below_top_k") or hi <= 0.01)]
            if chosen:
                c, p = rates(chosen)
                by_prob[label] = {"cuts": len(chosen), "control": c, "perturbed": p,
                                  "recovery": p / c if c else None}
        summary["regions"][region] = {
            "by_senior_probability_of_forced_token": by_prob,
            "cuts": len(rows), "control_success": control, "perturbed_success": perturbed,
            "recovery": perturbed / control if control else None,
            "se": dict(zip(("control", "perturbed", "recovery"), np.nanstd(draws, axis=0, ddof=1).tolist())),
            "by_alternative_rank": by_rank,
            "median_alt_prob": float(np.median(np.exp([i["alt_logprob"] for i in rows]))) if rows else None}
    common.save_json(out, {"summary": summary, "items": items})
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
