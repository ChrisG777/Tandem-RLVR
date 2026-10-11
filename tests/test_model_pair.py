"""Check separate junior provenance and all six paired evaluation phases."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT / 'train')]
import figure2
import select_figure2


class ModelPair(unittest.TestCase):
    def test_independent_selection_only_requires_the_local_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bases = {'base': {'path': '/senior'}, 'junior': {'path': '/junior'}}
            (root / 'base.json').write_text(json.dumps(bases))
            (root / 'grpo-full').mkdir()
            (root / 'grpo-full/initialization.json').write_text(json.dumps(bases))
            argv = ['select', '--root', str(root), '--out', str(root / 'models.json'), '--arm', 'grpo']
            with patch.object(sys, 'argv', argv), patch.object(select_figure2, 'verify_training', return_value={'path': '/selected'}) as verify:
                select_figure2.main()
            self.assertEqual(set(json.loads((root / 'models.json').read_text())), {'base', 'junior', 'grpo'})
            self.assertEqual([call.args for call in verify.call_args_list],
                             [(root / 'grpo-smoke', 'grpo', 3), (root / 'grpo-full', 'grpo', 200)])

    def test_selection_preserves_frozen_junior_and_checks_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bases = {'base': {'path': '/senior'}, 'junior': {'path': '/junior'}}
            (root / 'base.json').write_text(json.dumps(bases))
            for arm in ('grpo', 'tandem'):
                (root / f'{arm}-full').mkdir()
                (root / f'{arm}-full/initialization.json').write_text(json.dumps(bases))
            argv = ['select', '--root', str(root), '--out', str(root / 'models.json')]
            with patch.object(sys, 'argv', argv), patch.object(select_figure2, 'verify_training', return_value={'path': '/selected'}):
                select_figure2.main()
                result = json.loads((root / 'models.json').read_text())
                self.assertEqual(result['junior'], bases['junior'])
                self.assertEqual(set(result), {'base', 'junior', 'grpo', 'tandem'})
                (root / 'tandem-full/initialization.json').write_text(json.dumps({'base': bases['base']}))
                with self.assertRaisesRegex(ValueError, 'initialization'):
                    select_figure2.main()

    def test_paired_plot_reads_four_solo_and_two_handoff_results(self):
        calls = []
        def load(path):
            calls.append((path.parent.name, path.stem))
            return {'phase': path.stem, 'sampling': figure2.config.record()}
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['figure2', '--results', tmp, '--out', tmp + '/plot', '--paired-baselines']
            with patch.object(sys, 'argv', argv), patch.object(figure2.common, 'load_json', side_effect=load), \
                 patch.object(figure2.common, 'save_json'), patch.object(figure2, 'summarize', return_value={}), \
                 patch.object(figure2, 'plot'), patch.dict(figure2.COLORS), patch.dict(figure2.LABELS):
                figure2.main()
        self.assertEqual(set(calls), {(a, 'solo') for a in ('base', 'junior', 'grpo', 'tandem')} |
                         {(a, 'handoff') for a in ('grpo', 'tandem')})


if __name__ == '__main__':
    unittest.main()
