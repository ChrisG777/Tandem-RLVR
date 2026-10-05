"""Verify cancellation decisions without submitting or cancelling real jobs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slurm"))
import watch_figure2


class SmokeGuard(unittest.TestCase):
    def run_guard(self, state, valid):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if valid:
                for arm, phase in (("base", "solo"), ("grpo", "solo"),
                                   ("tandem", "solo"), ("grpo", "handoff"),
                                   ("tandem", "handoff")):
                    path = root / arm / f"{phase}.json"
                    path.parent.mkdir(exist_ok=True)
                    path.write_text(json.dumps({"phase": phase, "n": 2, "limit": 2,
                        "gens": [{"texts": ["answer", "answer"], "correct": [0, 1]}] * 2}))
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, f"10|{state}|0:0\n", "")

            argv = ["watch", "--smoke-job", "10", "--full-job", "20",
                    "--results", str(root), "--status", str(root / "status.json")]
            with patch.object(sys, "argv", argv), patch.object(watch_figure2.subprocess, "run", run), \
                    patch.object(watch_figure2.signal, "signal"):
                if state == "COMPLETED" and valid:
                    watch_figure2.main()
                else:
                    with self.assertRaises((RuntimeError, FileNotFoundError)):
                        watch_figure2.main()
            return calls, json.loads((root / "status.json").read_text())

    def test_successful_smoke_and_artifacts_leave_full_job_running(self):
        calls, status = self.run_guard("COMPLETED", True)
        self.assertNotIn(["scancel", "20"], calls)
        self.assertTrue(status["passed"])

    def test_failed_smoke_cancels_only_guarded_full_job(self):
        calls, status = self.run_guard("FAILED", False)
        self.assertEqual([c for c in calls if c[0] == "scancel"], [["scancel", "20"]])
        self.assertFalse(status["passed"])

    def test_success_exit_with_missing_artifacts_still_cancels_full_job(self):
        calls, status = self.run_guard("COMPLETED", False)
        self.assertIn(["scancel", "20"], calls)
        self.assertFalse(status["passed"])


if __name__ == "__main__":
    unittest.main()
