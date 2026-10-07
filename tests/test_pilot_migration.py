"""Verify revised calibration and host-side checkpoint continuation without GPUs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from check_shorthand import check_calibration


class PilotMigration(unittest.TestCase):
    def test_revised_budgets_keep_signal_and_completion_gates(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "data/manipulate_matrix"
            task.mkdir(parents=True)
            path = task / "calibration.parquet"
            pd.DataFrame([{"prompt": [{"role": "user", "content": "example"}]}]).to_parquet(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            (task / "manifest.json").write_text(json.dumps({"splits": {
                "calibration": {"sha256": digest, "rows": 1}}}))
            for budget, acc, completion in ((2048, .4, .47), (3072, .71, .9)):
                (root / f"manipulate_matrix-calibration-b{budget}.json").write_text(json.dumps({
                    "metrics": {"accuracy": acc, "answer_present": completion},
                    "provenance": {"max_tokens": budget, "model": "base", "data_sha256": digest}}))
            tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kw: [1])
            with patch.dict(sys.modules, {"transformers": SimpleNamespace(
                    AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: tokenizer))}):
                args = (root, root / "data", ["manipulate_matrix"], [2048, 3072])
                self.assertTrue(check_calibration(*args)["verified"])
                long_path = root / "manipulate_matrix-calibration-b3072.json"
                result = json.loads(long_path.read_text())
                result["metrics"]["answer_present"] = .2
                long_path.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "recalibration"):
                    check_calibration(*args)

    def test_walltime_signal_requires_complete_checkpoint(self):
        for complete in (True, False):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                actor = root / "train/manipulate_matrix-b3072/global_step_10/actor"
                actor.mkdir(parents=True)
                (actor.parent.parent / "latest_checkpointed_iteration.txt").write_text("10")
                (actor.parent / "data.pt").write_text("checkpoint")
                for name in ("model", "extra_state", "optim"):
                    if complete or name != "optim":
                        (actor / f"{name}_world_size_1_rank_0.pt").write_text("checkpoint")
                apptainer = root / "apptainer"
                apptainer.write_text('#!/bin/bash\nkill -USR1 "$PPID"\nsleep 0.1\n')
                apptainer.chmod(0o755)
                command = r'''
                    scontrol() { echo "$*" > "$RUN_ROOT/requeue-call"; }
                    export -f scontrol
                    source "$REPO/slurm/shorthand-runtime.sh"
                '''
                env = {**os.environ, "REPO": str(ROOT), "RUN_ROOT": str(root),
                       "TANDEM_CONTAINER_ACTIVE": "0",
                       "APPTAINER_IMAGE": "test.sif", "APPTAINER_BIN_DIR": str(root),
                       "CONTAINER_BIND": str(root), "PILOT_ENTRYPOINT": "shorthand-train.sbatch",
                       "PILOT_TASKS": "manipulate_matrix", "PILOT_BUDGETS": "2048 3072",
                       "SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "122",
                       "SLURM_ARRAY_TASK_ID": "1", "SLURM_RESTART_COUNT": "0"}
                proc = subprocess.run(["bash", "-euc", command], env=env, cwd=ROOT, capture_output=True, text=True)
                call = root / "requeue-call"
                self.assertEqual(call.exists(), complete, proc.stdout + proc.stderr)
                if complete:
                    self.assertEqual(call.read_text().strip(), "requeue 122_1")


if __name__ == "__main__":
    unittest.main()
