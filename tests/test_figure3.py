"""Scientific checks for marginal drift, JSD decomposition and trace validation."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
import figure3


class Labels:
    def decode(self, ids, **kwargs):
        return ['Alpha', 'Beta', 'Gamma', 'Delta'][ids[0]]


class Figure3Statistics(unittest.TestCase):
    def test_jsd_matches_library_and_bins_sum_even_with_zeros(self):
        counts = {'base': np.array([100000, 100, 1, 0]),
                  'grpo': np.array([80000, 2000, 0, 4]),
                  'tandem': np.array([90000, 400, 1, 2])}
        s = figure3.summarize(counts, Labels(), min_base_frequency=1e-8)
        for arm in ('grpo','tandem'):
            self.assertAlmostEqual(s['jsd'][arm], jensenshannon(counts['base'], counts[arm])**2)
            self.assertAlmostEqual(sum(s['jsd_by_base_frequency'][arm]), s['jsd'][arm])
            curve = s['survival'][arm]
            self.assertAlmostEqual(curve[0], 1)
            self.assertTrue(np.all(np.diff(curve) <= 1e-12))
        changed = figure3.summarize(counts, Labels(), pseudocount=1)
        self.assertEqual(s['jsd'], changed['jsd'])

    def test_identical_distributions_have_zero_divergence_and_drift(self):
        counts = {arm: np.array([100,10,2,1]) for arm in figure3.ARMS}
        s = figure3.summarize(counts, Labels())
        self.assertEqual(s['jsd']['grpo'], 0)
        self.assertEqual(s['jsd']['tandem'], 0)
        self.assertEqual(s['survival']['grpo'][1:], [0.]*300)
        self.assertEqual(s['scatter']['x'], [0.]*4)

    def test_reverse_direction_is_distinct_from_ending_closer(self):
        s = figure3._scatter_summary(np.array([1.,2.]), np.array([-3.,-1.]))
        self.assertEqual(s['opposite_direction_percent'], 100.)
        self.assertEqual(s['closer_percent'], 50.)

    def test_partial_and_duplicate_inputs_rejected(self):
        r = {'phase':'solo', 'n':32, 'sampling':figure3.config.record(),
             'gens':[{'set':'amc23_25','idx':0,'texts':['x']*32}]*2}
        with self.assertRaises(ValueError):
            figure3._validate(r)
        r['partial'] = True
        with self.assertRaises(ValueError):
            figure3._validate(r)


if __name__ == '__main__':
    unittest.main()
