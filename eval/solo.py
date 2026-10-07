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
    args = ap.parse_args()

    from transformers import AutoTokenizer

    sets = common.parse_sets(args.benchmarks)
    problems = common.load_problems(sets, args.limit)
    progress_path, progress = common.load_progress(args.out, {
        "phase": "solo", "model": args.model, "n": args.n,
        "sampling": config.record(), "seed": 17,
    }, problems)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = common.build_engine(args.model, args.gpu_util)

    gens = progress["gens"]
    for start in range(len(gens), len(problems), common.EVAL_BATCH_SIZE):
        batch = problems[start:start + common.EVAL_BATCH_SIZE]
        prompts = [common.chat_prefix(tok, p["content"]) for p in batch]
        outs = llm.generate(prompts, config.sampling_params(n=args.n, seed=17))
        for p, o in zip(batch, outs, strict=True):
            texts = [c.text for c in o.outputs]
            correct = [common.grade(t, p["gt"]) for t in texts]
            gens.append({"set": p["set"], "idx": p["idx"], "texts": texts, "correct": correct})
        common.save_json(progress_path, progress)
    counts = [int(sum(g["correct"])) for g in gens]

    result = {
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
