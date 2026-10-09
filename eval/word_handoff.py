"""Evaluate fixed seniors with the authors' native training-time tandem sampler."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import common
import config
from handoff import check_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--senior', required=True)
    parser.add_argument('--junior', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--decoding', choices=('training', 'evaluation'), default='training')
    args = parser.parse_args()
    evaluate(args.senior, args.junior, args.out, n=args.n, limit=args.limit,
             decoding=args.decoding)


def evaluate(senior: str, junior: str, out: Path, *, n: int = 8,
             limit: int = 0, decoding: str = 'training') -> dict:
    """Run GPU inference and atomically save complete batches plus a final JSON.

    Reuses only matching saved inputs. Word-boundary Bernoulli(0.5) redraws and
    the 32-token backstop come from the pinned patched vLLM, never Python switching.
    Requires one >=40-GiB compatible GPU and matching tokenizers. Reject missing
    authorship, illegal switches, incomplete batches, and mismatched restart inputs.
    Training decoding is 0.6/1/-1; evaluation uses the Figure 2 decoding constants.
    """
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    if n < 1 or limit < 0 or decoding not in ('training', 'evaluation'):
        raise ValueError('Invalid sample count, limit, or decoding preset')
    boundary_path = common.ROOT / 'train/assets/qwen3_word_boundary_ids.json'
    boundary_bytes = boundary_path.read_bytes()
    boundaries = set(json.loads(boundary_bytes))
    sampling = ({'temperature': 0.6, 'top_p': 1.0, 'top_k': -1, 'max_tokens': 3000}
                if decoding == 'training' else config.record())
    identity = {'phase': 'word-handoff', 'senior': senior, 'junior': junior,
                'n': n, 'sampling': sampling, 'decoding': decoding,
                'schedule': {'selection_strategy': 'word', 'prob_primary': 0.5,
                             'max_gap_tokens': 32, 'boundary_sha256': hashlib.sha256(boundary_bytes).hexdigest()},
                'seed_rule': '1000 * problem_index + sample_index',
                'max_model_len': common.MAX_MODEL_LEN,
                'budget_rule': 'min(3000, 4096 - prompt_tokens - 8)'}
    problems = common.load_problems(limit=limit)
    progress_path, progress = common.load_progress(str(out), identity, problems)
    if out.exists():
        result = common.load_json(out)
        if result.get('identity') != identity or len(result['gens']) != len(problems):
            raise ValueError('Existing result differs from requested evaluation')
        return result
    tokenizer = AutoTokenizer.from_pretrained(senior)
    junior_tokenizer = AutoTokenizer.from_pretrained(junior)
    check_vocab(tokenizer, junior_tokenizer, False)
    expected = {i for token, i in tokenizer.get_vocab().items() if token.startswith('Ġ')}
    if boundaries != expected:
        raise ValueError('Word boundaries differ from the selected tokenizer')
    prompts = [common.chat_prefix(tokenizer, p['content']) for p in problems]
    lengths = [len(tokenizer.encode(p)) for p in prompts]
    budgets = [common.response_budget(length) for length in lengths]
    if min(budgets) < 1:
        raise ValueError('A prompt exceeds the model context')
    engine = build_engine(senior, junior, boundary_path)
    gens = progress['gens']
    try:
        for start in range(len(gens), len(problems), common.EVAL_BATCH_SIZE):
            batch = problems[start:start + common.EVAL_BATCH_SIZE]
            repeated = [prompts[start+i] for i in range(len(batch)) for _ in range(n)]
            params = [SamplingParams(**{**sampling, 'max_tokens': budgets[start+i]},
                                     seed=1000*p['idx']+s)
                      for i,p in enumerate(batch) for s in range(n)]
            outputs = engine.generate(repeated, params, use_tqdm=True)
            if len(outputs) != len(batch)*n:
                raise ValueError('Incomplete generation batch')
            records = []
            for i, problem in enumerate(batch):
                samples = [record_completion(o.outputs[0], boundaries)
                           for o in outputs[i*n:(i+1)*n]]
                texts = [s.pop('text') for s in samples]
                records.append({'set': problem['set'], 'idx': problem['idx'],
                                'response_budget': budgets[start+i],
                                'texts': texts, 'correct': [common.grade(t, problem['gt']) for t in texts],
                                'samples': samples})
            gens.extend(records)
            common.save_json(progress_path, progress)
            print(f'WORD HANDOFF saved {len(gens)}/{len(problems)} problems', flush=True)
        samples = [s for row in gens for s in row['samples']]
        token_count = sum(len(s['token_ids']) for s in samples)
        senior_count = sum(sum(s['model_mask']) for s in samples)
        if not token_count or not 0.3 < senior_count/token_count < 0.7:
            raise ValueError('Authorship participation check failed')
        result = {**identity, 'identity': identity, 'benchmarks': list(common.DEFAULT_SETS),
                  'limit': limit, 'batch_size': common.EVAL_BATCH_SIZE,
                  'senior_token_fraction': senior_count/token_count,
                  'truncated_fraction': sum(s['finish_reason']=='length' for s in samples)/len(samples),
                  'metrics': common.metrics_by_set(problems, [int(sum(g['correct'])) for g in gens], n),
                  'gens': gens}
        common.save_json(out, result)
        print(json.dumps({'metrics': result['metrics']['macro'],
                          'senior_token_fraction': result['senior_token_fraction']}), flush=True)
        return result
    finally:
        engine.llm_engine.engine_core.shutdown()


def build_engine(senior: str, junior: str, boundaries: Path):
    """Load both BF16 models on GPU 0, with two bounded 5-GiB KV caches."""
    from vllm import LLM
    os.environ.pop('VLLM_TANDEM_CONFIG', None)
    os.environ.pop('VLLM_TANDEM_ALL_GPUS', None)
    return LLM(model=senior, dtype='bfloat16', enforce_eager=True,
               max_model_len=common.MAX_MODEL_LEN, max_num_seqs=8,
               max_num_batched_tokens=4096, gpu_memory_utilization=0.85,
               kv_cache_memory_bytes=5*1024**3, enable_prefix_caching=True,
               tandem_config={'enabled': True, 'frozen_model': junior,
                              'selection_strategy': 'word', 'prob_primary': 0.5,
                              'max_gap_tokens': 32, 'frozen_gpu_devices': [0],
                              'boundary_token_ids_path': str(boundaries)})


def record_completion(completion, boundaries: set[int]) -> dict:
    """Check every token's author and allowed switch boundary; preserve raw evidence."""
    tokens = list(completion.token_ids)
    mask = getattr(completion, 'tandem_model_mask', None)
    if not tokens or mask is None or len(mask) != len(tokens) or any(v not in (0,1) for v in mask):
        raise ValueError('Missing or malformed per-token tandem authorship')
    since = 0
    for i, token in enumerate(tokens):
        since += 1
        boundary = token in boundaries or since >= 32
        if i+1 < len(tokens) and mask[i+1] != mask[i] and not boundary:
            raise ValueError('Author switched outside a word boundary or gap backstop')
        if boundary:
            since = 0
    return {'text': completion.text, 'token_ids': tokens, 'model_mask': list(mask),
            'finish_reason': completion.finish_reason}


if __name__ == '__main__':
    main()
