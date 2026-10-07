"""Check that walltime continuation preserves job IDs and is bounded."""
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("continuation", Path(__file__).resolve().parents[1] / "slurm/continue-training.py")
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


class TrainingContinuation(unittest.TestCase):
    def test_requeues_saved_checkpoint_once_with_original_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actor = root / "global_step_40/actor"
            actor.mkdir(parents=True)
            (root / "latest_checkpointed_iteration.txt").write_text("40")
            (actor.parent / "data.pt").touch()
            for name in ("model", "optim", "extra_state"):
                (actor / f"{name}_world_size_1_rank_0.pt").touch()
            near = (datetime.now() + timedelta(minutes=2)).isoformat()
            outputs = iter([f"JobState=RUNNING Restarts=2 EndTime={near}",
                            f"JobState=RUNNING Restarts=3 EndTime={near}",
                            "JobState=COMPLETED Restarts=3"])
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(stdout=next(outputs) if command[1] == "show" else "")
            with patch.object(sys, "argv", ["continue-training.py", "--job", f"123={root}", "--state", str(root / "state.json")]), \
                 patch.object(continuation.subprocess, "run", side_effect=run), \
                 patch.object(continuation.time, "sleep"):
                continuation.main()
            self.assertEqual([c for c in calls if c[1] == "requeue"], [["scontrol", "requeue", "123"]])


if __name__ == "__main__":
    unittest.main()
