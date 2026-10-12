#!/usr/bin/env -S /data/scratch/chrisge/.local/bin/uv run --no-project --offline python
"""Run the existing CSAIL storage launcher with a separate solo-judge archive."""
from pathlib import Path
import os
import sys


def main() -> int:
    """Launch ephemeral grading with isolated history and unchanged Slurm guards.

    ELEGANCE_CODEX_HOME must name a private, initialized archive outside the repo.
    The installed launcher still owns node-local SQLite, requeue checks and backups.
    This judge never imports personal interactive transcripts into its archive.
    Authentication and the executable are provisioned separately, never committed.
    """
    archive = Path(os.environ["ELEGANCE_CODEX_HOME"]).resolve()
    shared = Path("/data/scratch/chrisge/.codex-cluster").resolve()
    if archive == shared or not archive.is_dir():
        raise RuntimeError("An initialized, independent judge archive is required")
    if archive.stat().st_uid != os.getuid() or archive.stat().st_mode & 0o077:
        raise RuntimeError("Judge archive must be private to the current user")
    if sys.argv[1:] != ["--version"] and not (
        sys.argv[1:2] == ["exec"] and "--ephemeral" in sys.argv[2:]
    ):
        raise RuntimeError("This runtime is restricted to ephemeral grading")
    sys.path.insert(0, "/data/scratch/chrisge/tools/codex-cluster")
    import codex_cluster
    codex_cluster.SHARED = archive
    codex_cluster.CONTROL = archive / "cluster-storage"
    codex_cluster.SNAPSHOTS = codex_cluster.CONTROL / "snapshots"
    codex_cluster.import_legacy_transcripts = lambda source, destination: None
    return codex_cluster.main()


if __name__ == "__main__":
    raise SystemExit(main())
