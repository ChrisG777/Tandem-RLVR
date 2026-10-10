"""APPS conversion, grading contracts, calibration gating and panel completeness."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "train")]
from experiments.coding.build_apps import prepare_splits
from experiments.coding.reward import normalize_tests, compute_score, extract_code
from experiments.coding.campaign import calibration


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [0] * (5000 if "LONG" in messages[0]["content"] else 30)


class CodingTests(unittest.TestCase):
    def test_test_conversion(self):
        tests = normalize_tests({"inputs": [[[1, 2], 3]], "outputs": [[4, 5]], "fn_name": "add"})
        self.assertEqual(tests["inputs"], ['[1, 2]\n3'])
        self.assertEqual(tests["outputs"], ['[4, 5]'])
        self.assertEqual(tests["fn_name"], "add")
        with self.assertRaises(ValueError):
            normalize_tests({"inputs": [], "outputs": []})
        with self.assertRaises(ValueError):
            normalize_tests({"inputs": ["x"], "outputs": []})

    def test_extraction_and_dispatch(self):
        self.assertEqual(extract_code('```python\nx=1\n```\n```python\nx=2\n```'), 'x=2\n')
        self.assertIsNone(extract_code('```python\nx=1'))
        gt = json.dumps(normalize_tests({"inputs": ["1"], "outputs": ["2"]}))
        self.assertEqual(compute_score("apps", "hello", gt)["format_error"], 1)
        with patch('experiments.coding.reward.run_sandbox', side_effect=RuntimeError("grader offline")):
            with self.assertRaisesRegex(RuntimeError, "grader offline"):
                compute_score("apps", '```python\nprint(2)\n```', gt)

    def test_splits(self):
        def row(i, tier, text=None):
            return {"problem_id": i, "difficulty": tier, "question": text or f"task {i}",
                    "starter_code": "", "input_output": '{"inputs":["1"],"outputs":["2"]}'}
        data = {"train": [row(i, "introductory" if i < 5 else "interview") for i in range(10)],
                "test": [row(20, "introductory", "task 0"), row(21, "interview", "LONG task")]}
        one, audit = prepare_splits(data, Tokenizer(), validation_per_tier=1)
        two, _ = prepare_splits(data, Tokenizer(), validation_per_tier=1)
        self.assertEqual(one, two)
        self.assertIn({"task_id": "train/0", "reason": "test_duplicate"}, audit)
        self.assertEqual(len(one['validation.parquet']), 2)
        self.assertTrue(one['test/apps_interview/test.parquet'][0]['extra_info']['overlength'])
        train_ids = {r['extra_info']['task_id'] for r in one['train.parquet']}
        self.assertFalse(train_ids & {r['extra_info']['task_id'] for r in one['validation.parquet']})

    def test_calibration_gate(self):
        rows = [{"set": "apps_introductory" if i < 100 else "apps_interview", "task_id": f"train/{i}",
                 "correct": [1,0]*4, "diagnostics": [{}]*8,
                 "samples": [{"n_tokens": 10, "finish_reason": "stop"}]*8} for i in range(200)]
        solo = {"phase": "solo", "n": 8, "sampling": {"temperature": .6, "top_p": 1., "top_k": -1, "max_tokens": 3000},
                "gens": rows, "dataset_sha256": "same", "grader": {}, "context": 8192, "max_prompt": 4096}
        from experiments.coding.build_apps import BASE_REVISION
        solo['model'] = '/snapshots/' + BASE_REVISION
        word = {**copy.deepcopy(solo), "phase": "word-handoff", "senior_token_fraction": .5,
                "senior": solo['model'], "junior": solo['model']}
        manifest = {"files": {f"validation/{tier}/test.parquet": {
            "task_ids": [r['task_id'] for r in rows if r['set'] == tier]}
            for tier in ('apps_introductory', 'apps_interview')}}
        kwargs = dict(manifest=manifest, manifest_sha256='same')
        self.assertTrue(calibration(solo, word, **kwargs)["passed"])
        wrong = copy.deepcopy(word)
        wrong['gens'][0]['task_id'] = 'test/0'
        with self.assertRaisesRegex(ValueError, 'validation manifest'):
            calibration(solo, wrong, **kwargs)
        for row in word['gens']:
            row['correct'] = [1]*8
        self.assertFalse(calibration(solo, word, **kwargs)["passed"])
        word['gens'].pop()
        with self.assertRaises(ValueError):
            calibration(solo, word, **kwargs)

    def test_resume_rejects_grader_change(self):
        import common
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d)/'out.json')
            problems = [{"set": "apps", "idx": 0}]
            path, progress = common.load_progress(out, {"grader": "a"}, problems)
            common.save_json(path, progress)
            with self.assertRaises(ValueError):
                common.load_progress(out, {"grader": "b"}, problems)

    def test_checkpoint_metric_is_parameterized(self):
        import figure2_checkpoint as checkpoint
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            records = {i: {"actor/pg_loss": 0., "val-core/apps/acc/best@4/mean": .5} for i in (1,2,3)}
            with patch.object(checkpoint, 'read_training_records', return_value=records), patch.object(checkpoint, 'verify_weights'):
                selected = checkpoint.verify_training(root, 'grpo', 3, validation_metric='val-core/apps/acc/best@4/mean')
                self.assertEqual(selected['step'], 1)
                with self.assertRaises(ValueError):
                    checkpoint.verify_training(root, 'grpo', 3)

    def test_panel_requires_exact_task_ids(self):
        import figure2
        saved = figure2.BENCHMARKS, figure2.TITLES, figure2.EXPECTED_SIZES, figure2.EXPECTED_IDS
        try:
            figure2.configure_panel({"benchmarks": {"apps_introductory": {"title": "Intro", "task_ids": ["test/1"]}}})
            result = {"phase": "solo", "n": 32, "gens": [{"set": "apps_introductory", "idx": 0, "task_id": "test/1", "correct": [1]*32}]}
            self.assertEqual(figure2.summarize(result, bootstrap=10)['macro']['mean'][0], 100.)
            result['gens'][0]['task_id'] = 'test/2'
            with self.assertRaisesRegex(ValueError, "task IDs"):
                figure2.summarize(result, bootstrap=10)
        finally:
            figure2.BENCHMARKS, figure2.TITLES, figure2.EXPECTED_SIZES, figure2.EXPECTED_IDS = saved

    def test_submission_gates_training_and_rejects_duplicate_campaign(self):
        from experiments.coding import submit
        calls = []
        def record(command, **kwargs):
            calls.append(command)
            return str(100 + len(calls)) + '\n'
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {'RUN_ROOT': tmp, 'REPO': str(ROOT)}), \
             patch.object(sys, 'argv', ['submit', '--cpu-args=--partition=cpu --mem=8G',
                                       '--eval-args=--partition=gpu --gpus=1 --mem=48G',
                                       '--train-args=--partition=gpu --mem=148G']), \
             patch.object(submit.subprocess, 'check_output', side_effect=record):
            submit.main()
            jobs = json.loads((Path(tmp)/'jobs.json').read_text())
            for arm in ('grpo', 'tandem'):
                smoke = next(c for c in calls if f'--job-name=apps-{arm}-smoke' in c)
                full = next(c for c in calls if f'--job-name=apps-{arm}-full' in c)
                self.assertIn('--dependency=afterok:' + jobs['gate'], smoke)
                self.assertIn('--dependency=afterok:' + jobs[arm+'-smoke'], full)
                self.assertIn('--kill-on-invalid-dep=yes', full)
                self.assertIn('--gpus=1', full)
            before = len(calls)
            with self.assertRaises(FileExistsError):
                submit.main()
            self.assertEqual(len(calls), before)


@unittest.skipUnless(sys.platform == "linux" and os.environ.get("APPS_GRADER_RUNTIME"), "Requires provisioned Linux sandbox")
class SandboxIntegration(unittest.TestCase):
    def test_real_candidate_outcomes(self):
        fixtures = [
            ("print(int(input())*2)", "pass"),
            ("print(0)", "wrong_answer"),
            ("def broken(:", "syntax_error"),
            ("raise ValueError('candidate')", "runtime_error"),
            ("while True: pass", "timeout"),
        ]
        gt = json.dumps(normalize_tests({"inputs": ["2\n"], "outputs": ["4\n"]}))
        for code, expected in fixtures:
            with self.subTest(expected=expected):
                actual = compute_score("apps", f"```python\n{code}\n```", gt)
                self.assertEqual(actual['acc'], float(expected == 'pass'))
                if expected != 'pass':
                    self.assertEqual(actual[expected], 1.)

    def test_real_isolation_and_call_based_execution(self):
        from experiments.coding.campaign import preflight
        preflight()
        gt = json.dumps(normalize_tests({"inputs": [[[1,2], 3]], "outputs": [[4,5]], "fn_name": "add"}))
        score = compute_score("apps", "```python\ndef add(xs, v):\n    return [x+v for x in xs]\n```", gt)
        self.assertEqual(score['acc'], 1.)

    def test_grader_startup_is_not_a_zero_reward(self):
        gt = json.dumps(normalize_tests({"inputs": ["2"], "outputs": ["4"]}))
        with patch.dict(os.environ, {"APPS_BWRAP": "/bin/false"}):
            with self.assertRaises(RuntimeError):
                compute_score("apps", "```python\nprint(4)\n```", gt)


if __name__ == "__main__":
    unittest.main()
