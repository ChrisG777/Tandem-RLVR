"""GPU check of optional FLA against Transformers' Qwen3.5 reference kernels."""
import unittest
from importlib.util import find_spec

import torch


@unittest.skipUnless(torch.cuda.is_available() and find_spec("fla"), "requires CUDA and optional FLA kernels")
class Qwen35Kernels(unittest.TestCase):
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
