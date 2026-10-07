import argparse
import sys

import common
import config

MAX_ROUNDS = 150
DEFAULT_JUNIOR = "Qwen/Qwen3-4B-Instruct-2507"

UTIL_PER_CARD = 0.85
UTIL_SHARED_CARD = 0.42

SENIOR = 0


def make_chains(problems, prompts, budgets, n):
    chains = []
    for p, base, budget in zip(problems, prompts, budgets):
        for s in range(n):
            chains.append(
                {
                    "p": p,
                    "base": base,
                    "budget": budget,
                    "text": "",
                    "turn": SENIOR,
                    "done": False,
                    "used": 0,
                    "seed": 1000 * p["idx"] + s,
                }
            )
    return chains


def advance(chain, seg):
    chain["text"] += seg.text
    chain["used"] += len(seg.token_ids)
    if chain["used"] >= chain["budget"] or seg.finish_reason == "length":
        chain["done"] = True
    elif seg.stop_reason is None:
        chain["done"] = True
    else:
        chain["turn"] = 1 - chain["turn"]


def run_turn(llm, active):
    prompts = [c["base"] + c["text"] for c in active]
    params = [
        config.sampling_params(
            seed=c["seed"] + c["used"],
            max_tokens=max(1, c["budget"] - c["used"]),
            stop=["\n\n"],
            include_stop_str_in_output=True,
        )
        for c in active
    ]
    outs = llm.generate(prompts, params, use_tqdm=False)
    for c, o in zip(active, outs):
        advance(c, o.outputs[0])


def run_all_rounds(engines, chains):
    for _ in range(MAX_ROUNDS):
        moved = False
        for turn, llm in enumerate(engines):
            active = [c for c in chains if not c["done"] and c["turn"] == turn]
            if active:
                run_turn(llm, active)
                moved = True
        if not moved:
            return


def build_engines(senior, junior, util, single_gpu):
    devices = common.visible_devices()
    if single_gpu:
        return [common.build_engine(m, util) for m in (senior, junior)]
    if len(devices) < 2:
        raise SystemExit(
            "handoff needs two visible GPUs so the senior and the junior are on "
            "separate cards. Set CUDA_VISIBLE_DEVICES to two ids, or pass "
            "--single-gpu to co-locate both engines on one card at a lower "
            f"memory fraction (default {UTIL_SHARED_CARD})."
        )
    return [
        common.build_engine(m, util, device=d)
        for m, d in ((senior, devices[0]), (junior, devices[1]))
    ]


def check_vocab(senior_tok, junior_tok, allow_mismatch):
    same = senior_tok.get_vocab() == junior_tok.get_vocab()
    print(
        f"[tok] senior vocab {len(senior_tok.get_vocab())} | "
        f"junior vocab {len(junior_tok.get_vocab())} | identical={same}"
    )
    if not same and not allow_mismatch:
        raise SystemExit(
            "senior and junior tokenizers differ; pass --allow-vocab-mismatch "
            "only if you have decided what that means for your pair."
        )
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senior", required=True)
    ap.add_argument("--junior", default=DEFAULT_JUNIOR, help="the frozen partner")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="chains per problem")
    ap.add_argument("--limit", type=int, default=0, help="0 = all problems")
    ap.add_argument("--benchmarks", default=",".join(common.DEFAULT_SETS))
    ap.add_argument("--gpu-util", type=float, default=None)
    ap.add_argument("--single-gpu", action="store_true",
                    default=len(common.visible_devices()) == 1,
                    help="both engines on one card (automatic with one visible GPU)")
    ap.add_argument("--allow-vocab-mismatch", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    util = args.gpu_util
    if util is None:
        util = UTIL_SHARED_CARD if args.single_gpu else UTIL_PER_CARD

    sets = common.parse_sets(args.benchmarks)
    problems = common.load_problems(sets, args.limit)
    tok = AutoTokenizer.from_pretrained(args.senior)
    jtok = tok if args.junior == args.senior else AutoTokenizer.from_pretrained(args.junior)
    same_vocab = check_vocab(tok, jtok, args.allow_vocab_mismatch)

    prompts = [common.chat_prefix(tok, p["content"]) for p in problems]
    budgets = [
        common.response_budget(max(len(tok(pr).input_ids), len(jtok(pr).input_ids)))
        for pr in prompts
    ]

    progress_path, progress = common.load_progress(args.out, {
        "phase": "handoff", "senior": args.senior, "junior": args.junior,
        "n": args.n, "sampling": config.record(), "max_rounds": MAX_ROUNDS,
        "same_vocab": same_vocab,
    }, problems)
    engines = build_engines(args.senior, args.junior, util, args.single_gpu)
    gens = progress["gens"]
    for start in range(len(gens), len(problems), common.EVAL_BATCH_SIZE):
        end = start + common.EVAL_BATCH_SIZE
        batch = problems[start:end]
        chains = make_chains(batch, prompts[start:end], budgets[start:end], args.n)
        run_all_rounds(engines, chains)
        progress["unfinished_chains"] += sum(not c["done"] for c in chains)
        for i, p in enumerate(batch):
            sample = chains[i * args.n:(i + 1) * args.n]
            gens.append({"set": p["set"], "idx": p["idx"],
                         "texts": [c["text"] for c in sample],
                         "correct": [common.grade(c["text"], p["gt"]) for c in sample]})
        common.save_json(progress_path, progress)

    unfinished = progress["unfinished_chains"]
    if unfinished:
        print(
            f"[warn] {unfinished}/{len(problems) * args.n} chains hit the MAX_ROUNDS={MAX_ROUNDS} "
            "cap before exhausting their token budget",
            file=sys.stderr,
        )

    counts = [int(sum(g["correct"])) for g in gens]

    result = {
        "phase": "handoff",
        "senior": args.senior,
        "junior": args.junior,
        "benchmarks": list(sets),
        "limit": args.limit,
        "n": args.n,
        "sampling": config.record(),
        "batch_size": common.EVAL_BATCH_SIZE,
        "max_rounds": MAX_ROUNDS,
        "unfinished_chains": unfinished,
        "same_vocab": same_vocab,
        "layout": "one engine per card" if not args.single_gpu else "both engines on one card",
        "metrics": common.metrics_by_set(problems, counts, args.n),
        "gens": gens,
    }
    common.save_json(args.out, result)
    print(f"HANDOFF {args.senior} + {args.junior}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in result["metrics"]["macro"].items()))


if __name__ == "__main__":
    main()
