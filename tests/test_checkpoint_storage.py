"""Exercise the checkpoint persistence function shipped in the verl patch."""
import errno
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


def patched_function():
    text = (Path(__file__).resolve().parents[1] / "third_party/verl-tandem.patch").read_text()
    block = text.split("+def persist_hf_checkpoint(", 1)[1].split(" class PPOTrainer", 1)[0]
    source = "def persist_hf_checkpoint(" + "\n".join(
        line[1:] for line in ("+" + block).splitlines() if line.startswith("+"))
    scope = {"os": os}
    exec(source, scope)
    return scope["persist_hf_checkpoint"]


class CheckpointStorage(unittest.TestCase):
    def test_rotation_keeps_weights_without_duplicate_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            actor = Path(directory) / "global_step_20/actor"
            source = actor / "huggingface/model.safetensors"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"immutable weights")
            destination = Path(patched_function()(str(actor), directory, 20)) / source.name
            self.assertEqual(source.stat().st_ino, destination.stat().st_ino)
            shutil.rmtree(actor.parent)
            self.assertEqual(destination.read_bytes(), b"immutable weights")

    def test_cross_filesystem_copy_and_failed_write_preserve_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            actor = Path(directory) / "global_step_20/actor"
            source = actor / "huggingface/model.safetensors"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"old")
            persist = patched_function()
            with patch("os.link", side_effect=OSError(errno.EXDEV, "different filesystem")):
                destination = Path(persist(str(actor), directory, 20)) / source.name
            self.assertNotEqual(source.stat().st_ino, destination.stat().st_ino)
            source.write_bytes(b"new")
            with patch("os.link", side_effect=OSError(errno.EDQUOT, "quota")):
                with self.assertRaises(OSError):
                    persist(str(actor), directory, 20)
            self.assertEqual(destination.read_bytes(), b"old")
            self.assertFalse(list(destination.parent.parent.glob(".step-*")))


if __name__ == "__main__":
    unittest.main()
