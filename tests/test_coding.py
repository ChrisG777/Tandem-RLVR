"""APPS conversion, grading contracts, calibration gating and panel completeness."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
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
    def test_exact_validation_pass4_and_completeness(self):
        import hashlib
        import pandas as pd
        from experiments.coding.validation import validation_pass4
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, root = base / 'data', base / 'grpo-smoke'
            data.mkdir()
            (root / 'validation').mkdir(parents=True)
            problems = [{'prompt': [{'content': f'<prompt-{i}>'}],
                         'extra_info': {'task_id': f'train/{i}'},
                         'reward_model': {'ground_truth': str(i)}} for i in range(200)]
            parquet = data / 'validation.parquet'
            pd.DataFrame(problems).to_parquet(parquet, index=False)
            manifest = {'files': {'validation.parquet': {
                'sha256': hashlib.sha256(parquet.read_bytes()).hexdigest(),
                'task_ids': [f'train/{i}' for i in range(200)]}}}
            raw = json.dumps(manifest)
            (data / 'manifest.json').write_text(raw)
            (root / 'data-manifest.json').write_text(raw)
            rows = [{'input': f'user\n<prompt-{i}>\nassistant\n', 'gts': str(i),
                     'step': 3, 'score': float(j == 0), 'acc': float(j == 0)}
                    for i in range(8) for j in range(4)]
            path = root / 'validation/3.jsonl'
            write = lambda values: path.write_text(''.join(json.dumps(r)+'\n' for r in values))
            write(rows)
            # With n=k=4, one passing sample gives exact pass@4=1, not ~0.68.
            self.assertEqual(validation_pass4(root, steps=3), {3: 1.0})
            rows[0]['score'] = rows[0]['acc'] = 0.0
            write(rows)
            self.assertEqual(validation_pass4(root, steps=3), {3: .875})
            write(rows[:-1])
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                validation_pass4(root, steps=3)
            rows[0]['gts'] = 'foreign test problem'
            write(rows)
            with self.assertRaisesRegex(ValueError, 'unique validation task'):
                validation_pass4(root, steps=3)
            parquet.write_bytes(parquet.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                validation_pass4(root, steps=3)

    def test_coding_selection_uses_exact_trace_scores(self):
        import figure2_checkpoint as checkpoint
        records = {i: {'actor/pg_loss': 0.} for i in range(1, 41)}
        records[20]['val-core/apps/acc/best@4/mean'] = .99
        records[40]['val-core/apps/acc/best@4/mean'] = .1
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(checkpoint, 'read_training_records', return_value=records), \
             patch.object(checkpoint, 'verify_weights'), \
             patch('experiments.coding.validation.validation_pass4', return_value={20: .5, 40: .8}) as score:
            root = Path(temporary)
            result = checkpoint.verify_training(root, 'grpo', 40, validation_metric='apps/validation/pass@4')
            self.assertEqual(result['step'], 40)
            score.return_value = {20: .8, 40: .8}
            self.assertEqual(checkpoint.verify_training(root, 'grpo', 40,
                             validation_metric='apps/validation/pass@4')['step'], 20)

    def test_regrading_preserves_and_resumes_saved_responses(self):
        import hashlib
        from experiments.coding import regrade as module
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'manifest.json').write_text('{}')
            problems = [{'set': 'apps_introductory', 'task_id': f'train/{i}',
                         'idx': i, 'gt': 'tests'} for i in range(2)]
            rows = [{**p, 'texts': ['saved response'], 'correct': [0],
                     'diagnostics': [{}], 'samples': [{'n_tokens': 3}]} for p in problems]
            source, out = root / 'source.json', root / 'regraded.json'
            original = {'dataset_sha256': hashlib.sha256(b'{}').hexdigest(),
                        'benchmarks': ['apps_introductory'], 'n': 1, 'grader': {'old': 1},
                        'gens': rows}
            source.write_text(json.dumps(original))
            with patch.object(module.common, 'load_problems', return_value=problems), \
                 patch.object(module.common, 'metrics_by_set', return_value={}), \
                 patch.object(module, 'grader_identity', return_value={'new': 2}), \
                 patch.object(module, 'compute_score', side_effect=[{'acc': 1}, RuntimeError('infra')]):
                with self.assertRaisesRegex(RuntimeError, 'infra'):
                    module.regrade(source, out, data=root, workers=1)
            with patch.object(module.common, 'load_problems', return_value=problems), \
                 patch.object(module.common, 'metrics_by_set', return_value={}), \
                 patch.object(module, 'grader_identity', return_value={'new': 2}), \
                 patch.object(module, 'compute_score', return_value={'acc': 1}) as score:
                result = module.regrade(source, out, data=root, workers=1)
                self.assertEqual(score.call_count, 1)
                self.assertEqual(result['regrading']['changed_rewards'], 2)
                self.assertEqual(result['gens'][0]['original_correct'], [0])
                self.assertEqual(result['gens'][0]['texts'], ['saved response'])
                self.assertEqual(json.loads(source.read_text()), original)
                source.write_text(source.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'differ'):
                    module.regrade(source, out, data=root, workers=1)

    def test_training_rejects_stale_calibration_grader(self):
        from experiments.coding.campaign import training_admission
        with tempfile.TemporaryDirectory() as root:
            report = Path(root) / "report.json"
            report.write_text(json.dumps({"passed": True, "grader": {"version": 1}}))
            with self.assertRaisesRegex(ValueError, "grader identity"):
                training_admission(report, expected_grader={"version": 2})
            self.assertFalse(training_admission(report, expected_grader={"version": 1})["provisional"])

    def test_checker_crash_before_candidate_is_infrastructure_failure(self):
        from experiments.coding.reward import run_sandbox

        def failed_checker(command, **kwargs):
            kwargs["stdout"].write(b'{"ready": true}\n')
            kwargs["stdout"].flush()
            return SimpleNamespace(returncode=139, wait=lambda **kwargs: None)

        with tempfile.TemporaryDirectory() as runtime:
            with patch.dict(os.environ, {"APPS_GRADER_RUNTIME": runtime}), \
                 patch('experiments.coding.reward.grader_identity', return_value={}), \
                 patch('experiments.coding.reward.subprocess.Popen', side_effect=failed_checker):
                with self.assertRaisesRegex(RuntimeError, 'Sandbox/checker exit 139'):
                    run_sandbox({"code": "pass", "tests": {}}, timeout=10)

    def test_provisional_training_admission(self):
        from experiments.coding.campaign import training_admission
        with tempfile.TemporaryDirectory() as d:
            report = Path(d) / 'report.json'
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                training_admission(report)
            self.assertTrue(training_admission(report, allow_pending=True)['provisional'])
            report.write_text('{"passed": false}')
            for allow in (False, True):
                with self.assertRaisesRegex(ValueError, 'failed'):
                    training_admission(report, allow_pending=allow)
            report.write_text('{"passed": true}')
            self.assertFalse(training_admission(report)['provisional'])
            report.write_text('invalid')
            with self.assertRaises(ValueError):
                training_admission(report, allow_pending=True)

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
        solo = {"phase": "solo", "n": 8, "sampling": {"temperature": .6, "top_p": 1., "top_k": -1, "max_tokens": 6000},
                "gens": rows, "dataset_sha256": "same", "grader": {}, "context": 12288, "max_prompt": 4096}
        from experiments.coding.build_apps import BASE_REVISION
        solo['model'] = '/snapshots/' + BASE_REVISION
        word = {**copy.deepcopy(solo), "phase": "word-handoff", "senior_token_fraction": .5,
                "senior": solo['model'], "junior": solo['model']}
        manifest = {"files": {f"validation/{tier}/test.parquet": {
            "task_ids": [r['task_id'] for r in rows if r['set'] == tier]}
            for tier in ('apps_introductory', 'apps_interview')}}
        kwargs = dict(manifest=manifest, manifest_sha256='same')
        self.assertTrue(calibration(solo, word, **kwargs)["passed"])
        boundary = copy.deepcopy(solo)
        for row in boundary['gens'][:20]:
            row['samples'] = [{'n_tokens': 6000, 'finish_reason': 'length'} for _ in range(8)]
        self.assertTrue(calibration(boundary, word, **kwargs)['passed'])
        boundary['gens'][20]['samples'][0] = {'n_tokens': 6000, 'finish_reason': 'length'}
        self.assertTrue(calibration(boundary, word, **kwargs)['passed'])
        boundary['gens'][20]['samples'][1] = {'n_tokens': 6000, 'finish_reason': 'length'}
        self.assertFalse(calibration(boundary, word, **kwargs)['passed'])
        legacy = copy.deepcopy(solo)
        legacy['sampling']['max_tokens'] = 3000
        legacy['context'] = 8192
        with self.assertRaises(ValueError):
            calibration(legacy, word, **kwargs)
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

    def test_response_budget_override_and_resume_identity(self):
        import argparse
        import common
        import config
        saved = common.DATA, common.MAX_MODEL_LEN, common.MAX_PROMPT, config.MAX_TOKENS
        try:
            parser = argparse.ArgumentParser()
            common.add_data_arguments(parser)
            common.configure(parser.parse_args([]))
            self.assertEqual(config.record()['max_tokens'], 3000)
            common.configure(parser.parse_args(['--context', '12288', '--max-prompt', '4096',
                                                '--max-response', '6000']))
            self.assertEqual(common.response_budget(4096), 6000)
            with tempfile.TemporaryDirectory() as d:
                out = str(Path(d) / 'result.json')
                problems = [{'set': 'apps', 'idx': 0}]
                path, progress = common.load_progress(out, {'sampling': config.record()}, problems)
                common.save_json(path, progress)
                config.MAX_TOKENS = 3000
                with self.assertRaises(ValueError):
                    common.load_progress(out, {'sampling': config.record()}, problems)
        finally:
            common.DATA, common.MAX_MODEL_LEN, common.MAX_PROMPT, config.MAX_TOKENS = saved

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
                self.assertIn('--export=ALL', full)
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
            ("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)", "runtime_error"),
            ("import os; os._exit(7)", "runtime_error"),
            ("import os; os._exit(0)", "runtime_error"),
            ("while True: pass", "timeout"),
            ("print('x' * (1024**2))", "runtime_error"),
            ("import sys; sys.stdout.write('x' * (1024**2)); raise ValueError('after full buffer')", "runtime_error"),
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
