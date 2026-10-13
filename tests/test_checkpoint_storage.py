"""Exercise the checkpoint persistence function shipped in the verl patch."""
import errno
import os
from pathlib import Path
import shutil
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def patched_function(name="persist_hf_checkpoint"):
    text = (Path(__file__).resolve().parents[1] / "third_party/verl-tandem.patch").read_text()
    block = text.split("+def persist_hf_checkpoint(", 1)[1].split(" class PPOTrainer", 1)[0]
    source = "def persist_hf_checkpoint(" + "\n".join(
        line[1:] for line in ("+" + block).splitlines() if line.startswith("+"))
    scope = {"os": os}
    exec(source, scope)
    return scope[name]


class CheckpointStorage(unittest.TestCase):
    def test_hf_export_interval_keeps_recovery_saves_independent(self):
        text = (Path(__file__).resolve().parents[1] / "third_party/verl-tandem.patch").read_text()
        section = text.split("diff --git a/verl/utils/checkpoint/checkpoint_manager.py", 1)[1]
        body = textwrap.dedent("\n".join(line[1:] for line in section.splitlines()
                                        if line.startswith("+") and not line.startswith("+++")))
        scope = {}
        exec("def should_export(self):\n" + textwrap.indent(body, "    "), scope)
        should_export = scope["should_export"]
        state = SimpleNamespace(checkpoint_config={"hf_model_interval": 20},
                                checkpoint_save_contents=["model", "optimizer", "extra", "hf_model"],
                                previous_global_step=0)
        exported = []
        for step in range(5, 201, 5):
            state.previous_global_step = step
            if should_export(state):
                exported.append(step)
            self.assertEqual(state.checkpoint_save_contents[:3], ["model", "optimizer", "extra"])
        self.assertEqual(exported, list(range(20, 201, 20)))
        state.checkpoint_config = {"hf_model_interval": 200}
        state.previous_global_step = 195
        self.assertFalse(should_export(state))
        state.previous_global_step = 200
        self.assertTrue(should_export(state))
        state.checkpoint_config = {}
        state.previous_global_step = None
        self.assertTrue(should_export(state))  # backward-compatible default
        state.checkpoint_config = {"hf_model_interval": 20}
        with self.assertRaises(RuntimeError):
            should_export(state)
        state.checkpoint_save_contents.remove("hf_model")
        self.assertFalse(should_export(state))
        for invalid in (0, -1, 1.5, True):
            state.checkpoint_config = {"hf_model_interval": invalid}
            with self.assertRaises(ValueError):
                should_export(state)

    def test_commit_keeps_only_latest_optimizer_and_preserves_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (10, 20, 30):  # 30 is an abandoned partial future save
                checkpoint = root / f"global_step_{step}"
                for role in ("actor", "critic"):
                    (checkpoint / role).mkdir(parents=True)
                    for rank in (0, 1):
                        (checkpoint / role / f"optim_world_size_2_rank_{rank}.pt").write_bytes(b"state")
                    (checkpoint / role / "model.pt").write_bytes(b"weights")
                if step != 30:
                    (checkpoint / "data.pt").write_bytes(b"loader")
            commit = patched_function("commit_latest_checkpoint")
            commit(directory, 20)
            commit(directory, 20)  # cleanup is idempotent
            self.assertEqual((root / "latest_checkpointed_iteration.txt").read_text(), "20")
            self.assertEqual(len(list(root.glob("global_step_*/**/optim_*.pt"))), 4)
            self.assertEqual(len(list(root.glob("global_step_*/**/model.pt"))), 6)
            self.assertTrue(all(p.is_file() for p in (root / "global_step_20/actor").glob("optim_*.pt")))

    def test_failed_commit_preserves_previous_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (10, 20):
                actor = root / f"global_step_{step}/actor"
                actor.mkdir(parents=True)
                (actor / "optim_world_size_1_rank_0.pt").write_bytes(b"state")
                (actor.parent / "data.pt").write_bytes(b"loader")
            marker = root / "latest_checkpointed_iteration.txt"
            marker.write_text("10")
            commit = patched_function("commit_latest_checkpoint")
            with patch("os.replace", side_effect=OSError(errno.EDQUOT, "quota")):
                with self.assertRaises(OSError):
                    commit(directory, 20)
            self.assertEqual(marker.read_text(), "10")
            self.assertEqual(len(list(root.glob("global_step_*/actor/optim_*.pt"))), 2)
            self.assertFalse(list(root.glob(".latest-*")))
            (root / "global_step_20/data.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "incomplete checkpoint"):
                commit(directory, 20)
            self.assertEqual(marker.read_text(), "10")

    def test_resume_adopts_old_states_but_not_future_partial_checkpoints(self):
        text = (Path(__file__).resolve().parents[1] / "third_party/verl-tandem.patch").read_text()
        section = text.split("diff --git a/verl/utils/checkpoint/fsdp_checkpoint_manager.py", 1)[1].split("diff --git ", 1)[0]
        added = textwrap.dedent("\n".join(line[1:] for line in section.splitlines()
                                        if line.startswith("+") and not line.startswith("+++")))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (10, 20, 60, 70):
                path = root / f"global_step_{step}"
                path.mkdir()
                (path / "data.pt").write_bytes(b"saved loader")
                if step != 20:  # a checkpoint whose actor was already rotated out
                    (path / "actor").mkdir()
            state = SimpleNamespace(rank=0)
            exec(added, {"self": state, "local_path": str(root / "global_step_60/actor")})
            self.assertEqual(state.previous_saved_paths,
                             [str(root / f"global_step_{step}/actor") for step in (10, 60)])

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
