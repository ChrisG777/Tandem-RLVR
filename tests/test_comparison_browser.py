"""Trace export must preserve text and never invent authorship."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('comparison_browser', Path(__file__).resolve().parents[1]/'eval/comparison_browser.py')
browser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(browser)


class Decoder:
    def decode(self, ids, skip_special_tokens=True):
        return ''.join({1: 'one ', 2: 'two', 3: ''}[i] for i in ids)


class TraceTests(unittest.TestCase):
    def row(self):
        return {'texts': ['one two'], 'correct': [1], 'samples': [
            {'token_ids': [1, 2, 3], 'model_mask': [1, 0, 0], 'finish_reason': 'stop'}]}

    def test_mask_means_senior_and_base_and_text_is_exact(self):
        result = browser.normalize_sample(self.row(), 0, Decoder())
        self.assertEqual(result['spans'], [{'author': 'senior', 'text': 'one '}, {'author': 'base', 'text': 'two'}])
        self.assertEqual(result['senior_fraction'], 1/3)
        self.assertEqual(''.join(s['text'] for s in result['spans']), result['text'])

    def test_invalid_mask_rejected(self):
        row = self.row()
        row['samples'][0]['model_mask'] = [1]
        with self.assertRaises(ValueError): browser.normalize_sample(row, 0, Decoder())

    def test_mismatch_keeps_original_without_colors(self):
        row = self.row()
        row['texts'] = ['different <script>text</script>']
        result = browser.normalize_sample(row, 0, Decoder())
        self.assertEqual(result['text'], row['texts'][0])
        self.assertNotIn('spans', result)
        self.assertIn('authorship_warning', result)

    def test_solo_has_no_invented_token_count(self):
        result = browser.normalize_sample({'texts': ['hello'], 'correct': [0]}, 0, Decoder())
        self.assertEqual(result, {'text': 'hello', 'correct': False})


if __name__ == '__main__':
    unittest.main()
