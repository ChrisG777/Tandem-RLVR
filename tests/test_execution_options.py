"""Execution switches must preserve experiment settings and fail closed on bad inputs."""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExecutionOptions(unittest.TestCase):
    def invoke(self, command):
        return subprocess.run(["bash", "-c", "set -eu; source train/execution.sh; " + command],
                              cwd=ROOT, text=True, capture_output=True)

    def test_baseline_preserves_caller_settings(self):
        result = self.invoke("export ENFORCE_EAGER=True OMP_NUM_THREADS=1; configure_training_execution qwen3 baseline; "
                             "test ${#EXECUTION_EXTRA[@]} = 0; test $ENFORCE_EAGER = True; test $OMP_NUM_THREADS = 1")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_optimized_keeps_objective_and_budget(self):
        result = self.invoke("export TRAIN_BATCH=16 MINI_BATCH=8 ROLLOUT_N=8 MAX_RESPONSE=6000 TOTAL_STEPS=200 LR=1e-6; "
                             "configure_training_execution qwen3 optimized; "
                             "test $TRAIN_BATCH:$MINI_BATCH:$ROLLOUT_N:$MAX_RESPONSE:$TOTAL_STEPS:$LR = 16:8:8:6000:200:1e-6; "
                             "printf '%s\\n' \"${EXECUTION_EXTRA[@]}\"")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in result.stdout.splitlines():
            self.assertFalse(option.startswith(("data.", "reward.", "trainer.")))

    def test_rejects_unknown_modes_and_missing_qwen_toolkit(self):
        for command in ["configure_training_execution unknown optimized", "configure_training_execution qwen3 unknown",
                        "CUDA_HOME=/nonexistent configure_training_execution qwen35 optimized"]:
            self.assertNotEqual(self.invoke(command).returncode, 0)


if __name__ == "__main__":
    unittest.main()
