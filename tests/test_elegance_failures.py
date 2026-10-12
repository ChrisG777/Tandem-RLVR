"""Judge process failures must retry or stop before a partial policy update."""
import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.elegance import judge


class JudgeFailures(unittest.TestCase):
    def tearDown(self):
        judge.codex_version.cache_clear()

    def test_version_probe_retries_and_caches_only_success(self):
        with patch.object(judge, 'run_codex', side_effect=[RuntimeError('timeout'), 'v1']) as run:
            with self.assertRaises(RuntimeError):
                judge.codex_version('codex', 300)
            self.assertEqual(judge.codex_version('codex', 300), 'v1')
            self.assertEqual(judge.codex_version('codex', 300), 'v1')
            self.assertEqual(run.call_count, 2)

    def test_timeout_kills_group_and_becomes_retryable(self):
        proc = Mock(pid=123)
        proc.communicate.side_effect = [subprocess.TimeoutExpired('codex', 1), ('', '')]
        with patch.object(judge.subprocess, 'Popen', return_value=proc), patch.object(judge.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                judge.run_codex(['codex'], None, 1)
            kill.assert_called_once_with(123, signal.SIGKILL)
            self.assertEqual(proc.communicate.call_count, 2)

    def test_exit_and_launch_errors_are_retryable(self):
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            judge.run_codex([sys.executable, '-c', 'raise SystemExit(7)'], None, 5)
        with self.assertRaisesRegex(RuntimeError, 'launch'):
            judge.run_codex(['/nonexistent/elegance-codex'], None, 5)

    def test_exhausted_retries_never_cache_a_grade(self):
        request = dict(question='Q', candidate='C', answer='A', references=[])
        with tempfile.TemporaryDirectory() as tmp, patch.object(judge, 'invoke_codex', side_effect=RuntimeError('timeout')) as invoke, patch.object(judge.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                judge.judge_solution(request, cache=Path(tmp))
            self.assertEqual(invoke.call_count, 3)
            self.assertEqual(list(Path(tmp).glob('*.json')), [])


class CompleteTraining(unittest.TestCase):
    def test_guard_before_padding_rejects_failed_or_missing_sessions(self):
        path = Path(__file__).resolve().parents[1] / 'third_party/verl/verl/trainer/ppo/v1/trainer_base.py'
        tree = ast.parse(path.read_text())
        guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and 'require_complete_training' in ast.unparse(n.test))
        code = compile(ast.Module(body=[guard], type_ignores=[]), str(path), 'exec')
        worker = SimpleNamespace(config=SimpleNamespace(
            trainer={'require_complete_training': True}, data=SimpleNamespace(train_batch_size=2),
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2))),
            replay_buffer=SimpleNamespace(failure_keys={'train': set()}))
        complete = ['a_0_0', 'a_1_0', 'b_0_0', 'b_1_0']
        for keys, failed, passes in ((complete, set(), True), (complete[:-1], set(), False),
                (complete[:2], set(), False), (complete, {'a'}, False),
                (complete[:-1] + ['b_2_0'], set(), False), ([], set(), False)):
            worker.replay_buffer.failure_keys['train'] = failed
            scope = dict(self=worker, batch=SimpleNamespace(keys=keys))
            with self.subTest(keys=keys, failed=failed):
                if passes:
                    exec(code, scope)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Incomplete training'):
                        exec(code, scope)


if __name__ == '__main__':
    unittest.main()
