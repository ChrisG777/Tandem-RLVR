"""Test 1: does junior authorship inside the answer region predict handoff failure?

Uses saved native word-handoff results (per-token `model_mask`, 1 = senior). For
APPS the answer region is the final graded ```python block; for math it is the
last \\boxed{...}. Effects are estimated within problems (problem-demeaned linear
probability model), so problem difficulty cannot drive them.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

CODE_BLOCK = re.compile(rb"(?m)^```python[ \t]*\r?\n(.*?)^```[ \t]*$", re.DOTALL)
FEATURES = ("junior_frac_answer", "junior_frac_reasoning", "switches_per_100_answer",
            "switches_per_100_reasoning")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="word-handoff result JSON")
    parser.add_argument("--domain", choices=("apps", "math"), required=True)
    parser.add_argument("--tokenizer", required=True, help="senior tokenizer path (shared vocabulary)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    result = json.loads(args.result.read_text())
    report = analyze(result, args.domain, AutoTokenizer.from_pretrained(args.tokenizer), args.bootstrap)
    report["source"] = str(args.result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("domain", "samples", "with_answer_region", "effects")}, indent=1))


def analyze(result: dict, domain: str, tokenizer, bootstrap: int = 500, seed: int = 0) -> dict:
    """Per-sample authorship features → within-problem effects on success.

    Returns effect sizes in success-probability points per +1 SD of each feature
    (others held fixed), with problem-bootstrap SEs. Samples without an answer
    region (format errors) are counted and excluded from the regression.
    """
    pieces = token_bytes(tokenizer)
    rows, missing = [], 0
    for problem in result["gens"]:
        for sample, correct in zip(problem["samples"], problem["correct"]):
            features = sample_features(sample["token_ids"], sample["model_mask"], pieces, domain)
            if features is None:
                missing += 1
                continue
            rows.append({"problem": problem["idx"], "correct": float(correct), **features})
    X = np.array([[r[f] for f in FEATURES] for r in rows])
    y = np.array([r["correct"] for r in rows])
    groups = np.array([r["problem"] for r in rows])
    effects = within_problem_effects(X, y, groups, bootstrap, seed)
    return {"domain": domain, "samples": len(rows) + missing, "with_answer_region": len(rows),
            "senior": result.get("senior"), "decoding": result.get("sampling"),
            "feature_means": {f: float(X[:, i].mean()) for i, f in enumerate(FEATURES)},
            "feature_sds": {f: float(X[:, i].std()) for i, f in enumerate(FEATURES)},
            "success_rate": float(y.mean()), "effects": effects}


def token_bytes(tokenizer) -> dict[int, bytes]:
    """Map every token id to its exact byte string (GPT-2 byte-level vocabularies)."""
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    decode = {c: b for b, c in bytes_to_unicode().items()}
    out = {}
    for token, index in tokenizer.get_vocab().items():
        try:
            out[index] = bytes(decode[c] for c in token)
        except KeyError:  # special/added tokens are not byte-level pieces
            out[index] = b""
    return out


def answer_span(text: bytes, domain: str) -> tuple[int, int] | None:
    """Byte span of the graded answer: last ```python body, or last \\boxed{...}."""
    if domain == "apps":
        blocks = list(CODE_BLOCK.finditer(text))
        return blocks[-1].span(1) if blocks else None
    start = text.rfind(b"\\boxed{")
    if start < 0:
        return None
    depth, i = 0, start + len(b"\\boxed")
    while i < len(text):
        depth += {ord("{"): 1, ord("}"): -1}.get(text[i], 0)
        if depth == 0:
            return start, i + 1
        i += 1
    return None


def sample_features(token_ids, mask, pieces, domain) -> dict | None:
    """Junior fraction and author switches inside vs before the answer region."""
    lengths = [len(pieces.get(t, b"")) for t in token_ids]
    span = answer_span(b"".join(pieces.get(t, b"") for t in token_ids), domain)
    if span is None:
        return None
    ends = np.cumsum(lengths)
    starts = ends - np.array(lengths)
    answer = (ends > span[0]) & (starts < span[1])
    reasoning = ends <= span[0]
    junior = 1 - np.asarray(mask, dtype=float)
    switch = np.r_[False, np.asarray(mask[1:]) != np.asarray(mask[:-1])]
    def stats(region):
        count = int(region.sum())
        return (float(junior[region].mean()) if count else 0.0,
                100 * float(switch[region].sum()) / count if count else 0.0)
    ja, sa = stats(answer)
    jr, sr = stats(reasoning)
    return {"junior_frac_answer": ja, "junior_frac_reasoning": jr,
            "switches_per_100_answer": sa, "switches_per_100_reasoning": sr,
            "answer_tokens": int(answer.sum()), "reasoning_tokens": int(reasoning.sum())}


def within_problem_effects(X, y, groups, bootstrap, seed) -> dict:
    """Problem-demeaned OLS; coefficients scaled to percentage points per +1 SD."""
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    problems = np.unique(groups)
    by_problem = {g: np.flatnonzero(groups == g) for g in problems}
    # Demeaning is within-problem, so resampled problems can reuse their blocks.
    Xd, yd = X / sd, y.copy()
    for index in by_problem.values():
        Xd[index] -= Xd[index].mean(axis=0)
        yd[index] -= yd[index].mean()
    def fit(index):
        return 100 * np.linalg.lstsq(Xd[index], yd[index], rcond=None)[0]
    estimate = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        chosen = rng.choice(problems, size=len(problems))
        draws.append(fit(np.concatenate([by_problem[g] for g in chosen])))
    se = np.std(draws, axis=0, ddof=1)
    return {f: {"pp_per_sd": float(e), "se": float(s)} for f, e, s in zip(FEATURES, estimate, se)}


if __name__ == "__main__":
    main()
