"""Run shell entrypoints against recording tools; never contact Slurm or install packages."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class JobWrappers(unittest.TestCase):
    def tool(self, directory: Path, name: str) -> Path:
        tool = directory / name
        tool.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys
path = pathlib.Path(os.environ["TOOL_CALLS"])
rows = path.read_text().splitlines() if path.exists() else []
with path.open("a") as handle:
    handle.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + "\\n")
if pathlib.Path(sys.argv[0]).name == "sbatch":
    print(100 + sum(json.loads(row)[0] == "sbatch" for row in rows))
''')
        tool.chmod(0o755)
        return tool

    def environment(self, directory: Path) -> dict:
        return {**os.environ, "REPO": str(ROOT), "RUN_ROOT": str(directory / "run"),
                "UV_BIN_DIR": str(directory), "TANDEM_ENV": str(directory / "train-env"),
                "EVAL_ENV": str(directory / "eval-env"), "DATA_ROOT": str(directory / "data"),
                "TOOL_CALLS": str(directory / "calls.jsonl"),
                "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                "APPTAINER_IMAGE": "", "TANDEM_CONTAINER_ACTIVE": "0"}

    def calls(self, directory: Path) -> list[list[str]]:
        return [json.loads(row) for row in (directory / "calls.jsonl").read_text().splitlines()]

    def test_pilot_dependency_chain_and_environment_survive_export_nil(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.tool(directory, "sbatch")
            env = {**self.environment(directory), "PILOT_TASKS": "manipulate_matrix ruletaker_shared",
                   "PILOT_BUDGETS": "2048 3072", "CALIBRATION_JOB": "90", "SETUP_JOB": "91",
                   "PILOT_LENGTH_PENALTY": "0.2", "CPU_SBATCH_ARGS": "--partition=cpu --mem=4G",
                   "TRAIN_SBATCH_ARGS": "--partition=train --gpus=1 --mem=144G",
                   "EVAL_SBATCH_ARGS": "--partition=eval --gpus=1 --mem=32G"}
            script = ROOT / "experiments/shorthand/submit.sh"
            proc = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            calls = self.calls(directory)
            self.assertEqual(len(calls), 6)
            for call, dependency in zip(calls, ["afterok:90:91", "afterok:100", "aftercorr:101",
                                               "afterok:100", "aftercorr:102", "afterok:103:104"]):
                self.assertIn("--dependency=" + dependency, call)
                self.assertIn("PILOT_TASKS=manipulate_matrix ruletaker_shared", call)
                self.assertIn("PILOT_LENGTH_PENALTY=0.2", call)
                entry = next(arg for arg in call if arg.endswith(".sbatch"))
                self.assertTrue((ROOT / entry).is_file())
            for i in (1, 2, 4):
                self.assertIn("--array=0-3", calls[i])
            self.assertIn("EXPECTED_COMPARISONS=16", calls[-1])
            again = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(len(self.calls(directory)), 6)

    def test_training_submission_passes_verifier_to_generic_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("sbatch", "scontrol"):
                self.tool(directory, name)
            env = {**self.environment(directory), "ARMS": "tandem", "HF_HOME": str(directory / "hf"),
                   "UV_CACHE_DIR": str(directory / "cache"), "WATCH_ENV": str(directory / "watch")}
            proc = subprocess.run(["bash", str(ROOT / "slurm/submit-training.sh")], env=env,
                                  text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            calls = self.calls(directory)
            monitor = next(call for call in calls if "slurm/watch-smoke.sbatch" in call)
            self.assertIn("--smoke-job", monitor)
            self.assertIn("train/figure2_checkpoint.py", monitor)
            self.assertIn("--full-job", monitor)
            self.assertTrue(any("Dependency=after:102" in call for call in calls))

    def test_setup_profiles_share_pinned_installation_without_preparing_datasets(self):
        for profile in ("inference", "training", "grading"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                self.tool(directory, "uv")
                env = {**self.environment(directory), "PROFILE": profile, "PREPARE_FORKS": "0"}
                proc = subprocess.run(["bash", str(ROOT / "slurm/setup.sh")], env=env,
                                      text=True, capture_output=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                calls = self.calls(directory)
                installs = [call for call in calls if call[1:3] == ["pip", "install"]]
                selected = "env/grading-requirements.txt" if profile == "grading" else "env/figure2-requirements.txt"
                self.assertTrue(any(selected in call and "env/requirements-freeze.txt" in call for call in installs))
                self.assertTrue(any("test_math_grading.py" in call for call in calls))
                self.assertEqual(any("third_party/verl" in call for call in installs), profile == "training")
                self.assertFalse(any("data/build_deepscaler.py" in call or "eval/download_figure2.py" in call for call in calls))
                self.assertTrue((directory / f"run/{profile}-environment.txt").is_file())

    def test_launchers_declare_exact_resume_paths_before_runtime(self):
        cases = [
            ("slurm/train-figure2.sbatch", {"ARM": "tandem", "MODE": "full"}, "RESUME_CHECKPOINT_ROOT", "tandem-full"),
            ("slurm/figure2.sbatch", {"MODE": "grpo-handoff"}, "RESUME_PROGRESS_FILE", "grpo/handoff.json.progress.json"),
            ("experiments/shorthand/slurm/shorthand-train.sbatch", {"SLURM_ARRAY_TASK_ID": "1", "PILOT_TASKS": "manipulate_matrix", "PILOT_BUDGETS": "2048 3072"}, "RESUME_CHECKPOINT_ROOT", "train/manipulate_matrix-b3072"),
            ("experiments/shorthand/slurm/shorthand-eval.sbatch", {"PHASE": "trained", "SLURM_ARRAY_TASK_ID": "1", "PILOT_TASKS": "manipulate_matrix", "PILOT_BUDGETS": "2048 3072"}, "RESUME_PROGRESS_DIR", "eval/manipulate_matrix-b3072"),
            ("experiments/drift/slurm/drift-train.sbatch", {"MODEL_KEY": "qwen"}, "RESUME_CHECKPOINT_ROOT", "qwen-lr1e-5-s42"),
        ]
        for script, overrides, variable, suffix in cases:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                env = {**self.environment(directory), "RESULTS_ROOT": str(directory / "run"), **overrides}
                prefix = (ROOT / script).read_text().split('source "$REPO/slurm/runtime.sh"')[0]
                proc = subprocess.run(["bash", "-c", prefix + '\nprintf "%s" "$' + variable + '"'],
                                      env=env, text=True, capture_output=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout, str(directory / "run" / suffix))


if __name__ == "__main__":
    unittest.main()
