"""Download immutable Figure 2 checkpoints and record their local snapshot paths."""
import argparse
import json
from pathlib import Path


def main() -> None:
    """Fetch pinned public weights into HF_HOME; atomically write --out paths JSON."""
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("figure2-models.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for spec in manifest.values():
        spec["path"] = snapshot_download(**spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
