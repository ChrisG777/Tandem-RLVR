"""Exercise the existing native Tandem backend before allocating training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Load pinned models on one GPU; atomically save verified authorship traces."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from experiments.word_handoff.word_handoff import build_engine, record_completion
    from eval import common

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--boundaries', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    models = json.loads(args.models.read_text())
    identity = {'models': models, 'boundaries': json.loads(args.boundaries.read_text())}
    if args.out.exists():
        if json.loads(args.out.read_text())['identity'] != identity:
            raise ValueError('Probe output belongs to a different model pair')
        return
    senior, junior = (models[key]['path'] for key in ('base', 'junior'))
    tok, jtok = (AutoTokenizer.from_pretrained(path) for path in (senior, junior))
    if tok.get_vocab() != jtok.get_vocab():
        raise ValueError('Native Tandem requires identical token vocabularies')
    boundaries = set(identity['boundaries'])
    if boundaries != {i for token, i in tok.get_vocab().items() if token.startswith('Ġ')}:
        raise ValueError('Word boundaries do not match this tokenizer')
    engine = build_engine(senior, junior, args.boundaries, gdn_prefill_backend='triton',
                          limit_mm_per_prompt={'image': 0, 'video': 0})
    try:
        prompt = common.chat_prefix(tok, 'Solve step by step: A shop sells 12 boxes of 8 pencils. How many pencils is that? Put the answer in \\boxed{}.')
        params = [SamplingParams(temperature=0.6, top_p=1, top_k=-1,
                                 max_tokens=256, seed=i) for i in range(4)]
        outputs = engine.generate([prompt] * 4, params, use_tqdm=False)
        samples = [record_completion(o.outputs[0], boundaries) for o in outputs]
        authors = {author for s in samples for author in s['model_mask']}
        if authors != {0, 1}:
            raise ValueError('Probe did not exercise both models')
        common.save_json(args.out, {'identity': identity, 'samples': samples})
        print('NATIVE PAIR PROBE PASSED', flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
