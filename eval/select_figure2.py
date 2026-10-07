"""Create an evaluation manifest from independently trained Figure 2 runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from figure2_checkpoint import verify_training


def main() -> None:
    """Require both complete 200-step runs and verified smoke tests; record local paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "base.json").read_text())
    if set(manifest) != {"base"}:
        raise ValueError("Initialization manifest must contain only the official base")
    for arm in ("grpo", "tandem"):
        root = args.root / f"{arm}-full"
        migration = root / "migration.json"
        if migration.exists():
            smoke = json.loads(migration.read_text())["source_smoke_verification"]
            if not smoke["verified"] or smoke["arm"] != arm or smoke["training_steps"] != 3:
                raise ValueError(f"Missing verified source smoke test for {arm}")
        else:
            verify_training(args.root / f"{arm}-smoke", arm, 3)
        if json.loads((root / "initialization.json").read_text()) != {"base": manifest["base"]}:
            raise ValueError(f"Unexpected initialization for {arm}")
        manifest[arm] = verify_training(root, arm, 200)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
