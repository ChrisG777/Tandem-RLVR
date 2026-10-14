"""Disposable native word-handoff throughput/trace qualification on one GPU."""
import argparse
import json
from pathlib import Path
import time

from experiments.word_handoff import word_handoff as word
import common
import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('eager2', 'graph2', 'graph8'), required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        print('Already completed', args.out, flush=True)
        return
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    root = args.root
    senior = json.loads((root/'grpo-full/selected.json').read_text())['path']
    junior = json.loads((root/'base.json').read_text())['base']['path']
    common.DATA = root/'data/test'
    common.MAX_MODEL_LEN, common.MAX_PROMPT, config.MAX_TOKENS = 12288, 4096, 6000
    problem = common.load_problems(('apps_introductory',), limit=1)[0]
    tokenizer = AutoTokenizer.from_pretrained(senior)
    prompt = common.chat_prefix(tokenizer, problem['content'])
    boundaries = word.ROOT/'train/assets/qwen3_word_boundary_ids.json'
    boundary_ids = set(json.loads(boundaries.read_text()))
    concurrency = 8 if args.mode == 'graph8' else 2
    cache = (30 if concurrency == 8 else 10)*1024**3
    kwargs = {} if args.mode == 'eager2' else {'compilation_config': {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY'}}
    start = time.monotonic()
    engine = word.build_engine(senior, junior, boundaries, max_num_seqs=concurrency,
                              kv_cache_memory_bytes=cache, enforce_eager=args.mode == 'eager2', **kwargs)
    load_seconds = time.monotonic()-start
    try:
        # Fixed-length throughput diagnostic, then an optional real-budget end-to-end test.
        engine.generate([prompt]*2, [SamplingParams(**{**config.record(), 'max_tokens': 64}, seed=i, ignore_eos=True) for i in range(2)], use_tqdm=False)
        params = [SamplingParams(**{**config.record(), 'max_tokens': 6000 if args.full else 512}, seed=i, ignore_eos=not args.full) for i in range(8)]
        start = time.monotonic()
        outputs = engine.generate([prompt]*8, params, use_tqdm=True)
        generation_seconds = time.monotonic()-start
        samples = [word.record_completion(o.outputs[0], boundary_ids) for o in outputs]
        total = sum(len(s['token_ids']) for s in samples)
        fraction = sum(sum(s['model_mask']) for s in samples)/total
        assert .3 < fraction < .7
        start = time.monotonic()
        grades = [common.grade_details(s['text'], problem['gt'], 'apps') for s in samples] if args.full else []
        result = {'mode': args.mode, 'full': args.full, 'load_seconds': load_seconds,
                  'generation_seconds': generation_seconds, 'grading_seconds': time.monotonic()-start,
                  'tokens': total, 'tokens_per_second': total/generation_seconds,
                  'senior_fraction': fraction, 'samples': samples, 'grades': grades}
        common.save_json(args.out, result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('samples','grades')}), flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
