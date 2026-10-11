"""GPU parity check for upstream FSDP2 deferred sync with CPU offload."""
import tempfile
import unittest

import torch
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class OffloadAccumulation(unittest.TestCase):
    def test_deferred_sync_preserves_gradient_and_update(self):
        with tempfile.TemporaryDirectory() as root:
            torch.distributed.init_process_group('nccl', init_method='file://'+root+'/init', rank=0, world_size=1)
            try:
                outputs = []
                for deferred in (False, True):
                    torch.manual_seed(42)
                    model = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 64)).cuda()
                    fully_shard(model, offload_policy=CPUOffloadPolicy(pin_memory=True),
                                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
                    for micro in range(4):
                        model.set_requires_gradient_sync(not deferred or micro == 3)
                        x = torch.randn(16, 128, device='cuda')
                        model(x).float().square().mean().div(4).backward()
                    grad = torch.cat([p.grad.to_local().flatten().clone() for p in model.parameters()])
                    optimizer.step()
                    weights = torch.cat([p.to_local().detach().flatten().clone() for p in model.parameters()])
                    outputs.append((grad, weights))
                    del optimizer, model
                old, new = outputs
                rms = (old[0]-new[0]).square().mean().sqrt()/old[0].square().mean().sqrt()
                delta = (old[1]-new[1]).abs().max()
                print(f'gradient relative RMS={rms.item():.6g}; max weight difference={delta.item():.6g}', flush=True)
                self.assertLess(rms.item(), 0.02)
                self.assertLess(delta.item(), 2e-7)
            finally:
                torch.distributed.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
