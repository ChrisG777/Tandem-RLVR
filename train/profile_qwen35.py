"""Opt-in timing hook for the existing verl FSDP engine; never changes its loss.

Set QWEN35_PROFILE_DIR and actor_rollout_ref.model.external_lib=train.profile_qwen35.
Writes per-process JSONL timings; QWEN35_PROFILE_TRACE=1 adds one backward trace. Import only in
isolated profiling runs: synchronization adds overhead to the measured stages.
"""
from contextlib import nullcontext
from functools import wraps
import json
import os
from pathlib import Path
import time


def install() -> None:
    """Instrument existing forward/backward/optimizer calls in this worker process."""
    directory = os.environ.get('QWEN35_PROFILE_DIR')
    if not directory:
        return
    import torch
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead

    if getattr(FSDPEngine, '_qwen35_timed', False):
        return
    FSDPEngine._qwen35_timed = True
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'timing-{os.getpid()}.jsonl'
    state = {'backward': 0, 'depth': 0}

    def record(stage: str, started: float, **extra) -> None:
        torch.cuda.synchronize()
        row = dict(stage=stage, seconds=time.perf_counter()-started,
                   threads=torch.get_num_threads(),
                   allocated_gib=torch.cuda.memory_allocated()/2**30,
                   peak_gib=torch.cuda.max_memory_allocated()/2**30, **extra)
        with path.open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print('QWEN35_TIMING '+json.dumps(row), flush=True)

    forward = FSDPEngineWithLMHead.forward_step
    @wraps(forward)
    def timed_forward(self, micro_batch, loss_function, forward_only):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = forward(self, micro_batch, loss_function, forward_only)
        ids = micro_batch['input_ids']
        record('forward_only' if forward_only else 'forward', started,
               examples=int(ids.shape[0]), tokens=int(ids.values().numel()) if ids.is_nested else ids.numel())
        return result
    FSDPEngineWithLMHead.forward_step = timed_forward

    backward = torch.Tensor.backward
    @wraps(backward)
    def timed_backward(self, *args, **kwargs):
        if state['depth']:
            return backward(self, *args, **kwargs)
        state['depth'] += 1
        state['backward'] += 1
        profile = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA])
                   if state['backward'] == 2 and os.environ.get('QWEN35_PROFILE_TRACE') == '1' else None)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with profile if profile is not None else nullcontext():
                result = backward(self, *args, **kwargs)
            record('backward', started, microbatch=state['backward'])
            if profile is not None:
                profile.export_chrome_trace(str(root/f'backward-{os.getpid()}.json'))
                for sort in ('self_cpu_time_total', 'self_cuda_time_total'):
                    (root/f'{sort}-{os.getpid()}.txt').write_text(profile.key_averages().table(sort_by=sort, row_limit=30))
            return result
        finally:
            state['depth'] -= 1
    torch.Tensor.backward = timed_backward

    optimizer = FSDPEngine.optimizer_step
    @wraps(optimizer)
    def timed_optimizer(self):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = optimizer(self)
        record('optimizer', started, grad_norm=result)
        return result
    FSDPEngine.optimizer_step = timed_optimizer


install()
