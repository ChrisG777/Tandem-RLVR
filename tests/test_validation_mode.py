"""Check validation request parameters and launcher isolation without model loading."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[1]
ENTRY_POINTS = (
    ("experimental/agent_loop/agent_loop.py", "AgentLoopWorker"),
    ("trainer/ppo/v1/agent_loop_tq.py", "AgentLoopWorkerTQ"),
)


def request_parameters(relative_path, class_name, validate):
    # Execute the installed method's parameter setup, stopping before Ray dispatch.
    spec = importlib.util.find_spec("verl")
    if spec is None:
        raise unittest.SkipTest("verl is not installed")
    path = Path(next(iter(spec.submodule_search_locations))) / relative_path
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "generate_sequences")
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.If)
               and isinstance(n.test, ast.Name) and n.test.id == "validate")
    module = ast.Module(body=method.body[:end + 1], type_ignores=[])
    config = SimpleNamespace(temperature=0.6, top_p=0.95, top_k=-1,
                             calculate_log_probs=False,
                             val_kwargs=SimpleNamespace(temperature=0.7, top_p=0.8, top_k=20))
    worker = SimpleNamespace(rollout_config=config, config=SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=config)))
    batch = ({"validate": validate} if class_name == "AgentLoopWorkerTQ"
             else SimpleNamespace(meta_info={"validate": validate}))
    scope = {"self": worker, "batch": batch, "Any": object}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["sampling_params"]


class ValidationParameters(unittest.TestCase):
    def test_validation_preserves_authorship_and_uses_validation_decoding(self):
        for path, cls in ENTRY_POINTS:
            with self.subTest(entry_point=cls):
                params = request_parameters(path, cls, validate=True)
                self.assertFalse(params.get("extra_args", {}).get("tandem_disabled", False))
                self.assertEqual((params["temperature"], params["top_p"], params["top_k"]),
                                 (0.7, 0.8, 20))

    def test_training_parameters_are_unchanged(self):
        for path, cls in ENTRY_POINTS:
            with self.subTest(entry_point=cls):
                params = request_parameters(path, cls, validate=False)
                self.assertFalse(params.get("extra_args", {}).get("tandem_disabled", False))
                self.assertEqual((params["temperature"], params["top_p"], params["top_k"]),
                                 (0.6, 0.95, -1))


class LauncherIsolation(unittest.TestCase):
    def test_each_arm_sets_its_own_rollout_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train"
            train.mkdir()
            binary = root / "bin"
            binary.mkdir()
            launchers = ("tandem_grpo.sh", "vanilla_grpo.sh", "grpo_g16.sh", "kl_reg.sh")
            for name in launchers:
                shutil.copy2(REPO / "train" / name, train / name)
            (train / "env.sh").write_text(
                "BASE_MODEL=test-model\nDATA_ROOT=/test-data\n"
                "CKPT_ROOT=/test-checkpoints\nPROJECT_NAME=test\nTANDEM_ENV_BIN=/test-env/bin\n")
            recorder = binary / "python3"
            recorder.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "with open(os.environ['VALIDATION_TEST_CAPTURE'], 'w') as f:\n"
                "    data = {k: os.environ.get(k) for k in "
                "['VLLM_TANDEM_CONFIG', 'VLLM_TANDEM_ALL_GPUS', 'RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES']}\n"
                "    data['argv'] = sys.argv[1:]\n    json.dump(data, f)\n")
            recorder.chmod(0o755)
            shutil.copy2(recorder, binary / "uv")
            capture = root / "environment.json"
            env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                   "VALIDATION_TEST_CAPTURE": str(capture), "KL_COEF": "0.001",
                   "VLLM_TANDEM_CONFIG": '{"enabled":true,"frozen_model":"stale-junior"}',
                   "VLLM_TANDEM_ALL_GPUS": "6,7"}
            env.pop("TANDEM_ENV_FILE", None)
            for name in launchers:
                with self.subTest(launcher=name):
                    result = subprocess.run(["bash", str(train / name)], env=env,
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual = json.loads(capture.read_text())
                    if name == "tandem_grpo.sh":
                        config = json.loads(actual["VLLM_TANDEM_CONFIG"])
                        self.assertTrue(config["enabled"])
                        self.assertEqual(config["frozen_model"], "test-model")
                    else:
                        self.assertIsNone(actual["VLLM_TANDEM_CONFIG"])
                        self.assertIsNone(actual["VLLM_TANDEM_ALL_GPUS"])
                    if name in ("tandem_grpo.sh", "vanilla_grpo.sh"):
                        batch_env = {**env, "TANDEM_ENV_FILE": str(REPO / "slurm/training-env.sh"),
                                     "TANDEM_ENV_BIN": "/test-env/bin", "BASE_MODEL": "test-model",
                                     "DATA_ROOT": "/test-data", "CKPT_ROOT": "/test-checkpoints",
                                     "PROJECT_NAME": "test"}
                        result = subprocess.run(["bash", str(train / name)], env=batch_env,
                                                text=True, capture_output=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        actual = json.loads(capture.read_text())
                        for arg in ("trainer.n_gpus_per_node=1", "data.dataloader_num_workers=2",
                                    "actor_rollout_ref.rollout.agent.num_workers=4",
                                    "transfer_queue.backend.SimpleStorage.num_data_storage_units=1",
                                    "data.train_batch_size=16", "actor_rollout_ref.rollout.n=8"):
                            self.assertIn(arg, actual["argv"])
                        self.assertEqual(actual["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"], "4294967296")


if __name__ == "__main__":
    unittest.main()
