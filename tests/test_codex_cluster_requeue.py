"""Site integration checks; CODEX_CLUSTER_SOURCE selects the installed launcher."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("CODEX_CLUSTER_SOURCE"), "requires the installed CSAIL launcher"
)
class RequeueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "cluster_storage", os.environ["CODEX_CLUSTER_SOURCE"]
        )
        cls.storage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.storage)

    def test_only_confirmed_restarted_single_node_is_accepted(self):
        fields = {
            "JobId": "12",
            "JobState": "RUNNING",
            "NumNodes": "1",
            "NodeList": "new",
            "BatchHost": "new",
            "Restarts": "1",
            "UserId": f"test({os.getuid()})",
        }

        def output(values):
            return " ".join(f"{k}={v}" for k, v in values.items())

        with patch.object(
            self.storage.subprocess, "check_output", return_value=output(fields)
        ):
            self.storage.verify_requeued_host("12", "new")
        for key, value in [
            ("JobId", "13"),
            ("JobState", "PENDING"),
            ("NumNodes", "2"),
            ("NodeList", "old"),
            ("BatchHost", "old"),
            ("Restarts", "0"),
            ("Restarts", "unknown"),
            ("UserId", "different(99999999)"),
        ]:
            with (
                self.subTest(key=key),
                patch.object(
                    self.storage.subprocess,
                    "check_output",
                    return_value=output(dict(fields, **{key: value})),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    self.storage.verify_requeued_host("12", "new")

    def test_confirmed_move_restores_and_keeps_exclusion(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / "control"
            control.mkdir()
            owner = control / "owner.json"
            owner.write_text(json.dumps({"job": "12", "host": "old"}))
            with (
                patch.object(self.storage, "CONTROL", control),
                patch.object(self.storage, "SNAPSHOTS", root / "snapshots"),
                patch.object(self.storage, "allocation_running", return_value=True),
                patch.object(self.storage, "verify_requeued_host") as confirm,
                patch.object(self.storage, "import_legacy_transcripts"),
                patch.object(
                    self.storage.subprocess, "check_output", return_value="ext4"
                ),
            ):
                self.storage.acquire_allocation("12", "new", root / "local")
                confirm.assert_called_once_with("12", "new")
                self.assertEqual(json.loads(owner.read_text())["host"], "new")
                with self.assertRaises(RuntimeError):
                    self.storage.acquire_allocation("13", "other", root / "other")
                self.assertEqual(json.loads(owner.read_text())["host"], "new")


if __name__ == "__main__":
    unittest.main()
