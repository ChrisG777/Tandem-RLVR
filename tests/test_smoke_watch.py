"""Exercise generic smoke cancellation and the separate Figure 2 artifact check."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("watch_smoke", ROOT / "slurm/watch-smoke.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)
sys.path.insert(0, str(ROOT / "eval"))
from check_figure2 import verify_outputs


class SmokeGuard(unittest.TestCase):
    def run_guard(self, state="COMPLETED", verifier_code=0, *, missing_state=False, cancel_code=0, verifier_error=None):
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "status.json"
            calls = []
            command = ["custom-verifier", "argument with spaces"]

            def run(argv, **kwargs):
                calls.append(argv)
                if argv[0] == "sacct":
                    return subprocess.CompletedProcess(argv, 0, "" if missing_state else f"10_2|{state}|0:0\n", "")
                if argv == command:
                    self.assertTrue(kwargs["check"])
                    self.assertEqual(kwargs["timeout"], 300)
                    if verifier_error:
                        raise verifier_error
                    if verifier_code:
                        raise subprocess.CalledProcessError(verifier_code, argv)
                return subprocess.CompletedProcess(argv, cancel_code if argv[0] == "scancel" else 0, "", "")

            argv = ["watch", "--smoke-job", "10_2", "--full-job", "20_2",
                    "--status", str(status_path), "--", *command]
            with patch.object(sys, "argv", argv), patch.object(watch.subprocess, "run", run), \
                    patch.object(watch.signal, "signal"), patch.object(watch.time, "sleep"):
                if state == "COMPLETED" and not verifier_code and not missing_state and verifier_error is None:
                    watch.main()
                else:
                    with self.assertRaises((RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                        watch.main()
            return calls, json.loads(status_path.read_text())

    def test_success_runs_exact_verifier_argv_without_cancelling(self):
        calls, record = self.run_guard()
        self.assertIn(["custom-verifier", "argument with spaces"], calls)
        self.assertFalse(any(c[0] == "scancel" for c in calls))
        self.assertTrue(record["passed"])

    def test_smoke_failure_cancels_only_guarded_job_without_verifier(self):
        calls, record = self.run_guard("FAILED")
        self.assertEqual([c for c in calls if c[0] == "scancel"], [["scancel", "20_2"]])
        self.assertFalse(any(c[0] == "custom-verifier" for c in calls))
        self.assertFalse(record["passed"])

    def test_verifier_failure_cancels(self):
        calls, record = self.run_guard(verifier_code=1)
        self.assertIn(["scancel", "20_2"], calls)
        self.assertFalse(record["passed"])

    def test_verifier_timeout_or_monitor_interruption_cancels(self):
        for error in (subprocess.TimeoutExpired("custom-verifier", 300), RuntimeError("Monitor interrupted by signal 15")):
            with self.subTest(error=type(error).__name__):
                calls, record = self.run_guard(verifier_error=error)
                self.assertIn(["scancel", "20_2"], calls)
                self.assertFalse(record["passed"])

    def test_scheduler_loss_cancels_after_five_attempts(self):
        calls, record = self.run_guard(missing_state=True)
        self.assertEqual(sum(c[0] == "sacct" for c in calls), 5)
        self.assertIn(["scancel", "20_2"], calls)
        self.assertFalse(record["passed"])

    def test_cancellation_failure_is_recorded(self):
        _, record = self.run_guard("FAILED", cancel_code=1)
        self.assertEqual(record["cancel_returncode"], 1)
        self.assertIn("cancel_error", record)

    def test_missing_verifier_or_same_job_rejected_without_scheduler_calls(self):
        for tail in ([], ["--", "true"]):
            with patch.object(sys, "argv", ["watch", "--smoke-job", "1", "--full-job", "1",
                                           "--status", "unused", *tail]), \
                    patch.object(watch.subprocess, "run") as run:
                with self.assertRaises(SystemExit):
                    watch.main()
                run.assert_not_called()

    def test_figure2_verifier_accepts_complete_and_rejects_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, phase in (("base", "solo"), ("grpo", "solo"), ("tandem", "solo"),
                               ("grpo", "handoff"), ("tandem", "handoff")):
                path = root / arm / f"{phase}.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({"phase": phase, "n": 2, "limit": 2,
                    "gens": [{"texts": ["answer", "answer"], "correct": [0, 1]}] * 2}))
            verify_outputs(root)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_outputs(root)


if __name__ == "__main__":
    unittest.main()
