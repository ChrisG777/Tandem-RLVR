"""Check the five Figure 2 smoke artifacts without running inference."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    verify_outputs(args.results)


def verify_outputs(root: Path) -> None:
    """Require all five smoke outputs, complete chains, and two binary grades/problem."""
    for arm, phase in (("base", "solo"), ("grpo", "solo"), ("tandem", "solo"),
                       ("grpo", "handoff"), ("tandem", "handoff")):
        path = root / arm / f"{phase}.json"
        result = json.loads(path.read_text())
        if (result["phase"] != phase or result["n"] != 2 or result["limit"] != 2
                or len(result["gens"]) != 2 or result.get("unfinished_chains", 0)
                or not result.get("same_vocab", True)):
            raise ValueError(f"Incomplete or incompatible smoke artifact: {path}")
        for row in result["gens"]:
            if (len(row["texts"]) != 2 or any(not text.strip() for text in row["texts"])
                    or len(row["correct"]) != 2
                    or any(grade not in (0, 1) for grade in row["correct"])):
                raise ValueError(f"Invalid smoke generations: {path}")


if __name__ == "__main__":
    main()
