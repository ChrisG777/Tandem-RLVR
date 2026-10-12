"""Exercise the existing native Tandem backend before allocating training."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Load pinned models on one GPU; atomically save verified authorship traces."""
    # Only this local diagnostic serializes its own worker callable.
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from experiments.word_handoff.word_handoff import build_engine, record_completion
    from eval import common

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--boundaries', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cuda-graphs', action='store_true', help='Exercise decode-only CUDA graphs without torch.compile')
    parser.add_argument('--long-rollouts', action='store_true', help='Grade 3000-token diagnostic rollouts, preserving text and authorship')
    parser.add_argument('--flash-attn-version', type=int, choices=(2, 3), help='Pin upstream attention implementation for a controlled comparison')
    parser.add_argument('--isolate-attention-metadata', action='store_true', help='Diagnostic: rebuild each attention group metadata instead of sharing it')
    args = parser.parse_args()
    models = json.loads(args.models.read_text())
    identity = {'models': models, 'boundaries': json.loads(args.boundaries.read_text())}
    if args.long_rollouts:
        identity['long_rollouts'] = True
    if args.flash_attn_version:
        identity['flash_attn_version'] = args.flash_attn_version
    if args.isolate_attention_metadata:
        identity['isolate_attention_metadata'] = True
    if args.cuda_graphs:
        identity['cuda_graphs'] = 'FULL_DECODE_ONLY'
        identity['torch_compile'] = False
    if args.out.exists():
        saved = json.loads(args.out.read_text())
        if saved['identity'] != identity:
            raise ValueError('Probe output belongs to a different model pair')
        audits = saved.get('weight_isolation', [])
        if audits and all(audit.get('attention_heads') for audit in audits):
            return
    senior, junior = (models[key]['path'] for key in ('base', 'junior'))
    tok, jtok = (AutoTokenizer.from_pretrained(path) for path in (senior, junior))
    if tok.get_vocab() != jtok.get_vocab():
        raise ValueError('Native Tandem requires identical token vocabularies')
    boundaries = set(identity['boundaries'])
    if boundaries != {i for token, i in tok.get_vocab().items() if token.startswith('Ġ')}:
        raise ValueError('Word boundaries do not match this tokenizer')
    question = 'Solve step by step: A shop sells 12 boxes of 8 pencils. How many pencils is that? Put the answer in \\boxed{}.'
    prompt = common.chat_prefix(tok, question)
    if prompt != common.chat_prefix(jtok, question) or tok(prompt).input_ids != jtok(prompt).input_ids:
        raise ValueError('Non-thinking prompt formats differ between the model pair')
    graph_kwargs = ({'compilation_config': {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY'}}
                    if args.cuda_graphs else {})
    if args.flash_attn_version:
        graph_kwargs['attention_config'] = {'flash_attn_version': args.flash_attn_version}
    engine = build_engine(senior, junior, args.boundaries, gdn_prefill_backend='triton',
                          limit_mm_per_prompt={'image': 0, 'video': 0},
                          enable_prefix_caching=False, enforce_eager=not args.cuda_graphs, **graph_kwargs)
    try:
        isolation = engine.collective_rpc(check_weight_isolation)
        if args.isolate_attention_metadata:
            engine.collective_rpc(isolate_attention_metadata)
        questions = [(question, '96')]
        if args.long_rollouts:
            questions = [
                ('What is the largest five-digit number whose digits add up to 20?', '99200'),
                ('What is the largest prime factor of 3328?', '13'),
                ('The bottom layer of a display has 28 cans, the next 25, and so on until one can at the top. How many cans are there in total?', '145'),
                ('Compute 2/(1-2/3).', '6'),
            ]
        prompts = [common.chat_prefix(tok, q + ' Solve step by step. Put your final answer inside \\boxed{}.')
                   for q, _ in questions for _ in range(4)] if args.long_rollouts else [prompt] * 4
        params = [SamplingParams(temperature=0.6, top_p=1, top_k=-1,
                                 max_tokens=3000 if args.long_rollouts else 256, seed=i)
                  for i in range(len(prompts))]
        outputs = engine.generate(prompts, params, use_tqdm=False)
        samples = [record_completion(o.outputs[0], boundaries) for o in outputs]
        if args.long_rollouts:
            from reward.math_boxed_reward import compute_score
            for i, sample in enumerate(samples):
                q, answer = questions[i // 4]
                sample.update(question=q, ground_truth=answer,
                              reward=compute_score('math', sample['text'], answer))
            print('LONG ROLLOUTS ' + json.dumps([
                {'tokens': len(s['token_ids']), 'reward': s['reward'], 'finish': s['finish_reason']}
                for s in samples]), flush=True)
        authors = {author for s in samples for author in s['model_mask']}
        if authors != {0, 1}:
            raise ValueError('Probe did not exercise both models')
        common.save_json(args.out, {'identity': identity, 'samples': samples,
                                    'weight_isolation': isolation})
        print('NATIVE PAIR PROBE PASSED', flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown()


def check_weight_isolation(worker) -> dict:
    """Audit the real vLLM loader: change/restore one senior weight, hash all junior weights."""
    import hashlib
    import torch

    runner = worker.model_runner
    junior = runner.tandem_manager.frozen_model
    heads = []
    layers = runner.vllm_config.compilation_config.static_forward_context
    for groups in runner.attn_groups:
        for group in groups:
            builder = group.get_metadata_builder()
            if not hasattr(builder, 'num_heads_q'):
                continue
            for name in group.layer_names:
                layer = layers[name]
                actual = (builder.num_heads_q, builder.num_heads_kv)
                expected = (layer.num_heads, layer.num_kv_heads)
                if actual != expected:
                    raise ValueError(f'Wrong attention metadata for {name}: {actual} != {expected}')
            heads.append({'layers': group.layer_names, 'query': actual[0], 'kv': actual[1]})

    def fingerprint() -> str:
        digest = hashlib.sha256()
        for name, value in junior.named_parameters():
            if value.requires_grad:
                raise ValueError(f'Junior parameter is trainable: {name}')
            digest.update(name.encode())
            digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()

    before = fingerprint()
    name, parameter = next((n, p) for n, p in runner.model.named_parameters()
                           if n.endswith('norm.weight') and p.ndim == 1)
    original = parameter.detach().clone()
    replacement = original + 1
    try:
        runner.model.load_weights([(name, replacement)])
        if not torch.equal(parameter, replacement):
            raise ValueError('Senior weight reload did not update the selected parameter')
        after = fingerprint()
        if before != after:
            raise ValueError('Senior weight reload changed frozen junior weights')
    finally:
        runner.model.load_weights([(name, original)])
    if not torch.equal(parameter, original):
        raise ValueError('Senior diagnostic weight was not restored')
    return {'senior_parameter': name, 'junior_sha256_before': before,
            'junior_sha256_after': after, 'senior_restored': True, 'attention_heads': heads}


def isolate_attention_metadata(worker) -> None:
    """Diagnostic only: force each builder to refresh its captured metadata buffers."""
    for groups in worker.model_runner.attn_groups:
        for group in groups:
            group.get_metadata_builder().supports_update_block_table = False


if __name__ == '__main__':
    main()
