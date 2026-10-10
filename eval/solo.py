import argparse

import common
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="senior checkpoint, hub id or local path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="samples per problem")
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--benchmarks", default=",".join(common.DEFAULT_SETS))
    common.add_data_arguments(ap)
    ap.add_argument("--decoding", choices=("evaluation", "training"), default="evaluation")
    args = ap.parse_args()
    common.configure(args)
    if args.decoding == "training":
        config.TEMPERATURE, config.TOP_P, config.TOP_K = 0.6, 1.0, -1

    from transformers import AutoTokenizer

    sets = common.parse_sets(args.benchmarks)
    problems = common.load_problems(sets, args.limit)
    progress_path, progress = common.load_progress(args.out, {
        **common.evaluation_identity(problems),
        "phase": "solo", "model": args.model, "n": args.n,
        "sampling": config.record(), "seed": 17,
    }, problems)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = common.build_engine(args.model, args.gpu_util)

    gens = progress["gens"]
    for start in range(len(gens), len(problems), common.EVAL_BATCH_SIZE):
        batch = problems[start:start + common.EVAL_BATCH_SIZE]
        prompts = [common.chat_prefix(tok, p["content"]) for p in batch]
        eligible = [not common.overlength(tok, pr) for pr in prompts]
        outs = iter(llm.generate([pr for pr, ok in zip(prompts, eligible) if ok], config.sampling_params(n=args.n, seed=17)) if any(eligible) else [])
        for p, ok in zip(batch, eligible):
            completions = next(outs).outputs if ok else []
            texts = [c.text for c in completions] if ok else [""] * args.n
            details = [common.grade_details(t, p["gt"], p.get("data_source", "math")) for t in texts] if ok else [{"acc": 0.0, "overlength": 1.0}] * args.n
            gens.append({"set": p["set"], "idx": p["idx"], "task_id": p.get("task_id"),
                         "texts": texts, "correct": [d["acc"] for d in details], "diagnostics": details,
                         "samples": [{"n_tokens": len(getattr(c, "token_ids", [])), "finish_reason": getattr(c, "finish_reason", None)} for c in completions],
                         "overlength": not ok})
        common.save_json(progress_path, progress)
    counts = [int(sum(g["correct"])) for g in gens]

    result = {
        **common.evaluation_identity(problems),
        "phase": "solo",
        "model": args.model,
        "benchmarks": list(sets),
        "limit": args.limit,
        "n": args.n,
        "sampling": config.record(),
        "batch_size": common.EVAL_BATCH_SIZE,
        "metrics": common.metrics_by_set(problems, counts, args.n),
        "gens": gens,
    }
    common.save_json(args.out, result)
    print(f"SOLO {args.model}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in result["metrics"]["macro"].items()))


if __name__ == "__main__":
    main()
