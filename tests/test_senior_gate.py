import types
import unittest

import torch

try:
    from verl.workers.utils.losses import _apply_tandem_senior_gate
    IMPORT_ERROR = None
except Exception as exc:
    _apply_tandem_senior_gate = None
    IMPORT_ERROR = exc


RESPONSE_MASK = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)

TANDEM_MASK = torch.tensor(
    [[1, 1, 0, 0, 1, 0, 1, 0],
     [0, 1, 1, 0, 1, 1, 0, 0]], dtype=torch.int64)


def gate(jr_tkn_weight=0.0, response_mask=None, tandem_mask=TANDEM_MASK):
    config = types.SimpleNamespace(tandem_jr_tkn_weight=jr_tkn_weight)
    data = {} if tandem_mask is None else {"tandem_model_mask": tandem_mask}
    metrics = {}
    mask = RESPONSE_MASK if response_mask is None else response_mask
    return _apply_tandem_senior_gate(config, mask, data, metrics), metrics


@unittest.skipIf(_apply_tandem_senior_gate is None,
                 f"verl not importable: {IMPORT_ERROR}")
class SeniorGate(unittest.TestCase):
    def test_junior_positions_are_zeroed(self):
        gated, _ = gate()
        self.assertEqual(gated.tolist(),
                         [[1, 1, 0, 0, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0, 0, 0]])

    def test_padding_stays_zero_even_where_the_mask_claims_the_senior(self):
        gated, _ = gate()
        self.assertEqual(gated[0, 6].item(), False)
        self.assertEqual(gated[1, 4].item(), False)

    def test_the_gate_only_removes(self):
        gated, _ = gate()
        self.assertTrue(torch.all(gated <= RESPONSE_MASK),
                        "the gate must not add positions to the response mask")

    def test_no_mask_leaves_the_response_mask_alone(self):
        gated, metrics = gate(tandem_mask=None)
        self.assertIs(gated, RESPONSE_MASK)
        self.assertEqual(metrics, {})

    def test_an_all_senior_mask_is_a_no_op(self):
        gated, _ = gate(tandem_mask=torch.ones_like(TANDEM_MASK))
        self.assertEqual(gated.tolist(), RESPONSE_MASK.tolist())

    def test_junior_weight_scales_instead_of_gating(self):
        gated, _ = gate(jr_tkn_weight=0.25)
        self.assertEqual(gated.dtype, torch.float32)
        self.assertEqual(gated.tolist(),
                         [[1.0, 1.0, 0.25, 0.25, 1.0, 0.25, 0.0, 0.0],
                          [0.25, 1.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0]])

    def test_reported_senior_fraction_is_over_the_ungated_mask(self):
        _, metrics = gate()
        self.assertAlmostEqual(
            metrics["actor/tandem_senior_token_frac"].aggregate(), 0.5)

    def test_senior_fraction_is_reported_under_the_junior_weight_path_too(self):
        _, metrics = gate(jr_tkn_weight=0.25)
        self.assertAlmostEqual(
            metrics["actor/tandem_senior_token_frac"].aggregate(), 0.5)


if __name__ == "__main__":
    unittest.main()
