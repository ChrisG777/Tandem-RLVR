"""Saved-output grading preserves evidence and rejects misaligned inputs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'eval'))
import common
from regrade import regrade


class Regrade(unittest.TestCase):
    def test_preserves_originals_and_cached_resume(self):
        problems = common.load_problems(('aime24',), 1)
        correct_text = r'\boxed{'+problems[0]['gt']+'}'
        data = {'phase': 'solo', 'n': 2, 'limit': 1, 'benchmarks': ['aime24'],
                'gens': [{'set': 'aime24', 'idx': 0, 'texts': [correct_text, 'no answer'], 'correct': [0, 0]}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out, cache = (root/n for n in ('source.json', 'out.json', 'cache.json'))
            source.write_text(json.dumps(data))
            result = regrade(source, out, cache, workers=1)
            self.assertEqual(result['gens'][0]['correct'], [1, 0])
            self.assertEqual(result['gens'][0]['original_correct'], [0, 0])
            self.assertEqual(json.loads(source.read_text()), data)
            with patch('regrade.grade_pair', side_effect=AssertionError('cached answers regenerated')):
                regrade(source, out, cache, workers=1)
            with self.assertRaisesRegex(ValueError, 'separate output'):
                regrade(source, source, cache)
            data['gens'][0]['idx'] = 2
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'ordering'):
                regrade(source, out, cache)


if __name__ == '__main__':
    unittest.main()
