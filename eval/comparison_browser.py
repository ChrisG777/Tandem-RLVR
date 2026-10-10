# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas==2.2.3", "pyarrow==19.0.1", "sympy==1.14.0", "pylatexenc==2.10", "tokenizers==0.21.4"]
# ///
"""Export saved evaluations to a static, local comparison browser; no inference."""
import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    ('base', 'Base solo', 'base/solo.json'),
    ('grpo', 'GRPO solo · step 160', 'grpo/solo.json'),
    ('tandem', 'Tandem solo · step 180', 'tandem/solo.json'),
    ('grpo-word', 'GRPO + base · word handoff', 'grpo/word-interim.json'),
    ('tandem-word', 'Tandem + base · word handoff', 'tandem/word-interim.json'),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/figure2-regraded')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/comparison-browser')
    parser.add_argument('--tokenizer', type=Path, default=ROOT/'outputs/comparison-browser-assets/tokenizer.json')
    parser.add_argument('--serve', action='store_true', help='Serve on localhost after export')
    parser.add_argument('--skip-build', action='store_true', help='Serve an existing export')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not args.skip_build:
        build_browser(args.results, args.out, args.tokenizer)
    if args.serve:
        handler = partial(SimpleHTTPRequestHandler, directory=str(args.out.resolve()))
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
        print(f'Browser: http://127.0.0.1:{args.port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


def build_browser(results: Path, out: Path, tokenizer_path: Path) -> dict:
    """Read five evaluation files and write index + per-problem JSON and UI assets.

    Preserve all samples and grades, validate dataset hashes/order, and reject
    duplicate/misaligned records. Uses disk proportional to input size. Sources
    are never modified. Tokenizer must match saved IDs; unverifiable span decoding
    falls back to the exact saved text with an explicit warning.
    """
    import common
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    out.mkdir(parents=True, exist_ok=True)
    (out/'problems').mkdir(exist_ok=True)
    catalog, conditions = {}, []
    for cid, label, relative in CONDITIONS:
        source = results/relative
        # Prefer a completed native-word artifact when it has been downloaded.
        if 'word' in cid and (results/relative.replace('word-interim', 'word-handoff')).exists():
            source = results/relative.replace('word-interim', 'word-handoff')
        result = json.loads(source.read_text())
        problems = common.problems_from_result(result)
        hashes = result.get('grading', {}).get('dataset_sha256', {})
        for name, expected in hashes.items():
            actual = hashlib.sha256((common.DATA/name/'test.parquet').read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Dataset hash mismatch: {name}')
        condition = {'id': cid, 'label': label, 'source': str(source.resolve()),
                     'partial': result.get('partial', False), 'n': result['n'],
                     'problems': len(result['gens']), 'model': result.get('model'),
                     'sampling': result.get('sampling'), 'grading': result.get('grading'),
                     'signature': result.get('signature')}
        conditions.append(condition)
        seen = set()
        for row in result['gens']:
            idx = row['idx']
            if not isinstance(idx, int) or idx < 0 or idx >= len(problems):
                raise ValueError(f'Invalid problem index: {source}: {idx}')
            problem = problems[idx]
            if row['set'] != problem['set'] or idx in seen:
                raise ValueError(f'Duplicate/misaligned problem: {source}: {idx}')
            seen.add(idx)
            # idx is global within each evaluation; canonical key uses dataset row.
            key = f"{problem['set']}-{problem['row']}"
            item = catalog.setdefault(key, {'id': key, 'benchmark': problem['set'],
                'row': problem['row'], 'prompt': problem['content'], 'answer': problem['gt'], 'conditions': {}})
            if (item['prompt'], item['answer']) != (problem['content'], problem['gt']):
                raise ValueError(f'Problem content mismatch: {key}')
            if len(row['texts']) != result['n'] or len(row['correct']) != result['n']:
                raise ValueError(f'Sample count mismatch: {key}')
            samples = [normalize_sample(row, i, tokenizer) for i in range(result['n'])]
            # Write each condition separately to keep exporter/browser memory bounded.
            write_json(out/'problems'/f'{key}-{cid}.json', samples)
            item['conditions'][cid] = {'count': len(samples), 'correct': sum(s['correct'] for s in samples)}
        print(f'{label}: {len(seen)} problems exported', flush=True)
    index = {'conditions': conditions, 'problems': list(catalog.values()),
             'note': 'Solo: temperature 0.7, top_p 0.8, top_k 20. Native word handoff uses training decoding (temperature 0.6, top_p 1, top_k -1); this is not a schedule-only ablation. Sample numbers identify independent draws, not paired random seeds.'}
    write_json(out/'index.json', index)
    for path in (ROOT/'ui/comparison').iterdir():
        if path.is_file():
            shutil.copy2(path, out/path.name)
    print(f"Exported {len(catalog)} problems to {out}", flush=True)
    return index


def normalize_sample(row: dict, index: int, tokenizer) -> dict:
    """Preserve source text/grade; attach verified author spans (1=senior, 0=base).

    Missing authorship remains unknown. Invalid masks fail; decoding mismatches
    retain plain source text rather than inventing character-level attribution.
    """
    text = row['texts'][index]
    sample = {'text': text, 'correct': bool(row['correct'][index])}
    if 'samples' not in row:
        return sample
    raw = row['samples'][index]
    ids, mask = raw['token_ids'], raw['model_mask']
    if len(ids) != len(mask) or any(v not in (0, 1) for v in mask):
        raise ValueError('Invalid authorship mask')
    sample.update(tokens=len(ids), finish_reason=raw.get('finish_reason'),
                  senior_fraction=sum(mask)/len(mask) if mask else None)
    spans = []
    start = 0
    for end in range(1, len(ids)+1):
        if end == len(ids) or mask[end] != mask[start]:
            spans.append({'author': 'senior' if mask[start] else 'base',
                          'text': tokenizer.decode(ids[start:end], skip_special_tokens=True)})
            start = end
    if ''.join(s['text'] for s in spans) != text and hasattr(tokenizer, 'id_to_token'):
        # The maintained decoder buffers UTF-8 characters split across tokens.
        from tokenizers.decoders import DecodeStream
        stream, pending, spans = DecodeStream(skip_special_tokens=True), set(), []
        for token, author in zip(ids, mask):
            pending.add(author)
            piece = stream.step(tokenizer, token)
            if piece:
                name = 'mixed' if len(pending) > 1 else ('senior' if author else 'base')
                if spans and spans[-1]['author'] == name:
                    spans[-1]['text'] += piece
                else:
                    spans.append({'author': name, 'text': piece})
                pending.clear()
    if ''.join(s['text'] for s in spans) == text:
        sample['spans'] = spans
    else:
        sample['authorship_warning'] = 'Token spans could not be decoded losslessly; showing exact saved text without colors.'
    return sample


def write_json(path: Path, value: object) -> None:
    """Atomically replace one generated JSON file."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
    temporary.replace(path)


if __name__ == '__main__':
    main()
