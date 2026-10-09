"""Build compact RuleTaker-derived deduction and object Re-ARC pilot splits."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import pandas as pd
from problog import get_evaluatable
from problog.program import PrologString
from transformers import AutoTokenizer
from reasoning_gym.arc.rearc_utils import generators, verifiers

from build_shorthand import REVISION, SPLITS

RULETAKER_REVISION = "abaacec9364992eff5ec4555b837e20fee2f2ff0"
TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
TOKENIZER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
FAMILIES = ("00d62c1b", "6d75e8bb", "a1570a43", "d364b489")
TASKS = ("ruletaker_shared", "rearc_objects")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ruletaker-source", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.task, args.out_dir, args.seed, args.ruletaker_source), indent=2))


def build_dataset(task: str, out_dir: Path, seed: int = 42,
                  ruletaker_source: Path | None = None) -> dict[str, Any]:
    """Write immutable disjoint splits with oracle checks and <=1024-token prompts.

    RuleTaker supplies records/rendering, ProbLog supplies entailment, and pinned
    Reasoning Gym supplies Re-ARC generators/verifiers. Never show oracle metadata
    or family IDs to the model. Refuse overwrites and upstream revision drift.
    """
    import importlib.metadata
    origin = json.loads(importlib.metadata.distribution("reasoning-gym").read_text("direct_url.json"))
    if origin["vcs_info"]["commit_id"] != REVISION:
        raise ValueError("Use env/shorthand-data with its pinned Reasoning Gym")
    rt = load_ruletaker(ruletaker_source) if task == "ruletaker_shared" else None
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"task": task, "seed": seed, "revision": REVISION,
                "ruletaker_revision": RULETAKER_REVISION if rt else None,
                "tokenizer": TOKENIZER, "tokenizer_revision": TOKENIZER_REVISION,
                "max_prompt_tokens": 1024, "families": list(FAMILIES) if not rt else [],
                "oracle": "ProbLog 2.2.10" if rt else "paired Re-ARC verifier",
                "splits": {}}
    seen_prompts, seen_instances = set(), set()
    for split_no, (split, count) in enumerate(SPLITS.items()):
        source_seed = seed + 1_000_000 * (split_no + 1)
        rows = []
        attempts = 0
        while len(rows) < count and attempts < count * 100:
            index = attempts
            attempts += 1
            rng = random.Random(source_seed + index)
            if rt:
                # Fix the target pattern before rejection sampling: every block
                # of 16 accepted theories has exactly half true query labels.
                truths = [bool((len(rows) % 16) & (1 << bit)) for bit in range(4)]
                entry = rule_instance(rng, rt, longer=split == "long", truths=truths)
            else:
                family = FAMILIES[len(rows) % len(FAMILIES)]
                entry = arc_instance(rng, family, longer=split == "long")
            if entry is None:
                continue
            content, answer, meta, instance_keys = entry
            digest = sha(content)
            if digest in seen_prompts or any(key in seen_instances for key in instance_keys):
                continue
            length = len(tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                        tokenize=True, add_generation_prompt=True))
            if length > 1024:
                continue
            seen_prompts.add(digest)
            seen_instances.update(instance_keys)
            meta["instance_keys"] = instance_keys
            rows.append({"data_source": task, "prompt": [{"role": "user", "content": content}],
                         "ability": "deduction" if rt else "induction",
                         "reward_model": {"ground_truth": answer, "style": "rule"},
                         "extra_info": {"index": len(rows), "split": split,
                                        "source_seed": source_seed, "source_index": index,
                                        "prompt_sha256": digest, "prompt_tokens": length,
                                        "family": meta.get("family", "shared_deduction"),
                                        "oracle_json": json.dumps(meta, sort_keys=True)}})
        if len(rows) != count:
            raise RuntimeError(f"Insufficient eligible instances: {task}/{split}: {len(rows)}")
        path = out_dir / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        manifest["splits"][split] = {"rows": count, "seed": source_seed, "examined": attempts,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "max_prompt_tokens": max(r["extra_info"]["prompt_tokens"] for r in rows)}
        print(f"{task}/{split}: {count} rows, {attempts} candidates", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_ruletaker(source: Path | None):
    """Load only upstream data records; require an unmodified pinned checkout."""
    if source is None:
        raise ValueError("--ruletaker-source must name the pinned allenai/ruletaker checkout")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--", "common.py"], text=True)
    if commit != RULETAKER_REVISION or dirty:
        raise ValueError("RuleTaker revision mismatch or modified common.py")
    spec = importlib.util.spec_from_file_location("upstream_ruletaker_records", source / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rule_instance(rng: random.Random, rt, *, longer: bool = False,
                  truths: list[bool] | None = None) -> tuple | None:
    """Four balanced entailment queries sharing a branching rule graph; no proofs in prompt."""
    names = rng.sample(["ada", "ben", "clee", "dana", "eli", "faye", "gus", "hana", "ivan", "june"],
                       7 if longer else 5)
    predicates = rng.sample(["red", "blue", "green", "round", "rough", "kind", "quiet", "young",
                             "warm", "large", "small", "brave", "bright", "calm", "happy", "strong"], 11)
    relations = rng.sample(["likes", "visits", "follows", "helps"], 2)
    def atom(i, *args):
        return rt.Fact("+", predicates[i], list(args))
    def edge(i, *args):
        return rt.Fact("+", relations[i], list(args))
    rules = [rt.Rule([atom(0, "X"), atom(1, "X")], atom(5, "X")),
             rt.Rule([atom(2, "X"), atom(3, "X")], atom(6, "X")),
             rt.Rule([atom(5, "X"), edge(0, "X", "Y"), atom(6, "Y")], atom(7, "X")),
             rt.Rule([atom(7, "X"), atom(4, "X")], atom(8, "X")),
             rt.Rule([atom(5, "X"), edge(1, "X", "Y"), atom(7, "Y")], atom(9, "X")),
             rt.Rule([atom(6, "X"), atom(9, "X")], atom(10, "X")),
             rt.Rule([atom(8, "X"), edge(0, "X", "Y")], atom(9, "Y")),
             rt.Rule([atom(10, "X")], atom(8, "X"))]
    facts = [atom(i, name) for name in names for i in range(5) if rng.random() < 0.65]
    facts += [edge(i, name, other) for name in names for other in names if name != other
              for i in range(2) if rng.random() < 0.25]
    theory = rt.Theory(facts, rules)
    queries = [atom(i, name) for i in range(7, 11) for name in names]
    # These theories are deterministic. Removing probability-one annotations
    # avoids compiling gratuitous random variables. False fallback clauses
    # declare possibly empty base predicates without adding facts to the theory.
    program = theory.program("problog").replace("1.0::", "")
    program += "\n" + "\n".join(f"{p}(X) :- fail." for p in predicates[:5])
    program += "\n" + "\n".join(f"{r}(X,Y) :- fail." for r in relations)
    program += "\n" + "\n".join(q.logical_form("problog", is_assertion=True) for q in queries)
    result = {str(key): value for key, value in get_evaluatable().create_from(PrologString(program)).evaluate().items()}
    selected, answers = [], []
    for i in range(7, 11):
        truth = truths[i - 7] if truths is not None else rng.choice([False, True])
        candidates = [q for q in queries if q.predicate == predicates[i]
                      and bool(result[q.logical_form("problog", standalone=False).replace(", ", ",")]) == truth]
        if not candidates:
            return None
        selected.append(rng.choice(candidates))
        answers.append("true" if truth else "false")
    # Permuting questions prevents output position from identifying proof depth.
    order = rng.sample(range(4), 4)
    selected, answers = [selected[i] for i in order], [answers[i] for i in order]
    sentences = [f.nl() for f in facts] + [r.nl() for r in rules]
    rng.shuffle(sentences)
    prompt = ("Use only the facts and rules below. Variables X and Y can each denote any named entity. "
              "Rules apply repeatedly until no new facts follow. A statement is true if it follows; "
              "otherwise answer false.\n\n" + "\n".join(sentences) + "\n\nQuestions:\n" +
              "\n".join(f"{i + 1}. {q.nl()}" for i, q in enumerate(selected)) +
              "\nSolve step by step. End with <answer> followed by exactly four lowercase true/false "
              "values in question order, separated by spaces, then </answer>.")
    key = sha(json.dumps(sorted(str(f) for f in facts) + sorted(str(r) for r in rules)))
    return prompt, " ".join(answers), {"theory": theory.to_json(), "queries": [q.to_json() for q in selected],
                                       "program": program, "answers": answers, "entities": len(names)}, [key]


def arc_instance(rng: random.Random, family: str, *, longer: bool = False) -> tuple | None:
    """Three demonstrations and one query from one hidden object-rule family."""
    generator = getattr(generators, f"generate_{family}")
    verifier = getattr(verifiers, f"verify_{family}")
    examples, keys = [], []
    for position in range(4):
        for _ in range(300):
            entry = generator(rng, 0.0, 0.3)
            grid = entry["input"]
            max_side = max(len(grid), len(grid[0]))
            lower, upper = (8, 9) if longer and position == 3 else (5, 7)
            if not lower <= max_side <= upper or min(len(grid), len(grid[0])) < 4:
                continue
            if entry["input"] == entry["output"] or verifier(grid) != entry["output"]:
                continue
            key = sha(json.dumps([family, entry["input"]]))
            if key in keys:
                continue
            examples.append(entry)
            keys.append(key)
            break
        else:
            return None
    prompt = "Infer the same grid transformation from all three examples and apply it to the final input.\n"
    for i, entry in enumerate(examples[:3]):
        prompt += f"\nExample {i + 1} input:\n{board(entry['input'])}\nOutput:\n{board(entry['output'])}\n"
    prompt += (f"\nFinal input:\n{board(examples[3]['input'])}\nSolve step by step. End with the output "
               "grid between <answer> and </answer>, one row per line, digits separated by spaces. "
               "Inside those tags write only the grid, without commentary or a code block.")
    return prompt, board(examples[3]["output"]), {"family": family, "examples": examples}, keys


def board(grid) -> str:
    return "\n".join(" ".join(map(str, row)) for row in grid)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    main()
