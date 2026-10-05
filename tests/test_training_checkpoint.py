"""Exercise training integrity and validation-only checkpoint selection."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from figure2_checkpoint import VALIDATION, verify_training


class TrainingCheckpoint(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for step, score in ((1, 0.4), (2, 0.6), (3, 0.5)):
            data = {"actor/pg_loss": 0.01, "actor/tandem_senior_token_frac": 0.5,
                    VALIDATION: score}
            rows.append({"step": step, "data": data})
            path = root / "hf" / f"global_step_{step}"
            path.mkdir(parents=True)
            (path / "config.json").write_text('{"model_type":"qwen3"}')
            header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
        self.write_rows(root, rows)
        return rows

    def write_rows(self, root, rows):
        (root / "metrics-1-0.jsonl").write_text("\n".join(map(json.dumps, rows)))

    def test_selects_best_validation_instead_of_last_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertEqual(verify_training(root, "tandem", 3)["step"], 2)

    def test_missing_mask_fails_tandem_but_not_grpo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.fixture(root)
            del rows[1]["data"]["actor/tandem_senior_token_frac"]
            self.write_rows(root, rows)
            with self.assertRaises(ValueError):
                verify_training(root, "tandem", 3)
            verify_training(root, "grpo", 3)

    def test_truncated_weights_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            shard = root / "hf/global_step_2/model.safetensors"
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                verify_training(root, "tandem", 3)


if __name__ == "__main__":
    unittest.main()
