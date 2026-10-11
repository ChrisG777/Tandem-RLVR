"""GPU check of optional FLA against Transformers' Qwen3.5 reference kernels."""
import unittest
from importlib.util import find_spec

import torch


@unittest.skipUnless(torch.cuda.is_available() and find_spec("fla"), "requires CUDA and optional FLA kernels")
class Qwen35Kernels(unittest.TestCase):
    def test_large_vocabulary_cross_entropy(self):
        from fla.modules.fused_cross_entropy import cross_entropy_loss

        torch.manual_seed(42)
        logits = torch.randn(128, 248320, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(logits.shape[-1], (len(logits),), device="cuda")
        reference = logits.float().requires_grad_()
        expected = reference.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        expected.mean().backward()
        for inplace in (False, True):
            actual_input = logits.clone().requires_grad_()
            actual = -cross_entropy_loss(actual_input, labels, inplace_backward=inplace)[0]
            actual.mean().backward()
            error = ((actual_input.grad.float() - reference.grad).square().mean().sqrt()
                     / reference.grad.square().mean().sqrt())
            print(f"cross entropy inplace={inplace}: log-prob max error="
                  f"{(actual-expected).abs().max().item():.6g}, gradient relative RMS={error.item():.6g}", flush=True)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
            self.assertLess(error.item(), 0.01)

    def test_batched_forward_and_backward(self):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

        torch.manual_seed(42)
        shape = (2, 129, 2, 128)
        inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        inputs += [-torch.rand(shape[:-1], device="cuda"),
                   torch.rand(shape[:-1], device="cuda", dtype=torch.bfloat16)]
        upstream = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        results = []
        for kernel in (torch_chunk_gated_delta_rule, chunk_gated_delta_rule):
            arguments = [x.detach().clone().requires_grad_() for x in inputs]
            output, _ = kernel(*arguments, use_qk_l2norm_in_kernel=True)
            (output.float() * upstream.float()).sum().backward()
            results.append([output, *(x.grad for x in arguments)])
        # BF16 fused kernels reorder arithmetic. Check normalized RMS error,
        # including gate/decay gradients and two independent recurrent states.
        for label, reference, actual in zip(("output", "q", "k", "v", "g", "beta"), *results):
            self.assertTrue(torch.isfinite(actual).all(), label)
            error = ((actual.float() - reference.float()).square().mean().sqrt()
                     / reference.float().square().mean().sqrt().clamp_min(1e-8))
            print(f"{label}: relative RMS error={error.item():.6g}", flush=True)
            self.assertLess(error.item(), 0.02, label)


if __name__ == "__main__":
    unittest.main()
