"""Exercise training integrity and validation-only checkpoint selection."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from figure2_checkpoint import VALIDATION, verify_training, record_resume_gap, read_training_records


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

    def gap_fixture(self, root):
        rows = self.fixture(root)
        self.write_rows(root, rows[:1])
        (root / "metrics-1-1.jsonl").write_text(json.dumps(rows[2]) + "\n")
        # Cluster filesystems can give successive attempts identical timestamps.
        for path in root.glob("metrics-*.jsonl"):
            os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        actor = root.resolve() / "global_step_2" / "actor"
        log = root / "resume.log"
        log.write_text("\n".join(
            f"[Rank 0] Loaded {component} from {actor}/{filename}_world_size_1_rank_0.pt"
            for component, filename in (("model", "model"), ("optimizer", "optim"),
                                        ("rng", "extra_state"), ("lr_scheduler", "extra_state"))))
        return log

    def test_resume_gap_requires_evidence_and_excludes_missing_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = self.gap_fixture(root)
            with self.assertRaisesRegex(ValueError, "no resume evidence"):
                verify_training(root, "tandem", 3)
            record_resume_gap(root, 2, log)
            selected = verify_training(root, "tandem", 3)
            self.assertEqual(selected["unobserved_metric_steps"], [2])
            self.assertEqual(selected["step"], 3)
            self.assertNotIn(2, read_training_records(root))

    def test_model_only_resume_log_is_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = self.gap_fixture(root)
            log.write_text(log.read_text().splitlines()[0])
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint-load"):
                record_resume_gap(root, 2, log)

    def test_resume_evidence_requires_intact_weights_and_correct_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = self.gap_fixture(root)
            record_resume_gap(root, 2, log)
            evidence_path = root / "resume-evidence/step-2.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["root"] = "/another/run"
            evidence_path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "another run"):
                verify_training(root, "tandem", 3)
            shard = root / "hf/global_step_2/model.safetensors"
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Truncated checkpoint"):
                record_resume_gap(root, 2, log)

    def test_retry_discards_abandoned_steps_and_stale_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "metrics-1-1.jsonl").write_text(json.dumps({
                "step": 2, "data": {"actor/pg_loss": 0.02}}) + "\n")
            records = read_training_records(root)
            self.assertEqual(set(records), {1, 2})
            self.assertNotIn(VALIDATION, records[2])
            with self.assertRaises(ValueError):
                verify_training(root, "grpo", 3)

    def test_missing_final_validation_still_fails_after_valid_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = self.gap_fixture(root)
            record_resume_gap(root, 2, log)
            path = root / "metrics-1-1.jsonl"
            row = json.loads(path.read_text())
            del row["data"][VALIDATION]
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Final held-out validation"):
                verify_training(root, "tandem", 3)


if __name__ == "__main__":
    unittest.main()
