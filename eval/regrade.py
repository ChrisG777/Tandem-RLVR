"""Regrade saved benchmark generations on CPUs, preserving source scores and traces."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import common
from hendrycks_math_grader import extract_answer, _try_import_math_verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--phase', choices=('solo', 'handoff', 'word-handoff'))
    parser.add_argument('--n', type=int)
    args = parser.parse_args()
    regrade(args.input, args.out, args.cache, workers=args.workers,
            phase=args.phase, n=args.n)


def regrade(source: Path, out: Path, cache_path: Path, *, workers: int = 4,
            phase: str | None = None, n: int | None = None) -> dict:
    """CPU-only rescore; never overwrite the input or discard original grades.

    Validate benchmark/sample alignment. Cache distinct answer/reference pairs,
    keyed by grader source and pinned dependencies; checkpoint every 128 pairs.
    Partial generation snapshots stay explicitly partial. No inference is run.
    """
    if source.resolve() == out.resolve() or workers < 1:
        raise ValueError('Use a separate output and at least one CPU worker')
    _try_import_math_verify()
    backend = {p: version(p) for p in ('math-verify', 'latex2sympy2_extended',
                                      'sympy', 'antlr4-python3-runtime', 'pylatexenc')}
    expected = {'math-verify': '0.9.0', 'latex2sympy2_extended': '1.11.0',
                'sympy': '1.14.0', 'antlr4-python3-runtime': '4.9.3', 'pylatexenc': '2.10'}
    if backend != expected:
        raise RuntimeError(f'Unexpected grading dependencies: {backend}')
    files = [common.ROOT/'reward'/f for f in ('hendrycks_math_grader.py', 'math_boxed_reward.py')]
    grader_hash = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    identity = {'grader_sha256': grader_hash, 'dependencies': backend}
    cache = common.load_json(cache_path) if cache_path.exists() else {'identity': identity, 'grades': {}}
    if cache['identity'] != identity:
        raise ValueError('Grader changed; use a fresh grading cache')
    raw = source.read_bytes()
    result = json.loads(raw)
    result['phase'] = result.get('phase', phase)
    result['n'] = result.get('n', n)
    if not result['phase'] or not isinstance(result['n'], int) or result['n'] < 1:
        raise ValueError('Progress snapshots require --phase and --n')
    sets = tuple(result.get('benchmarks', common.DEFAULT_SETS))
    problems = common.load_problems(sets, result.get('limit', 0))
    rows = result['gens']
    if not rows or len(rows) > len(problems):
        raise ValueError('Invalid problem count')
    pairs, row_keys = {}, []
    for row, problem in zip(rows, problems):
        if (row['idx'], row['set']) != (problem['idx'], problem['set']):
            raise ValueError('Benchmark ordering differs from saved generations')
        if len(row['texts']) != result['n'] or len(row['correct']) != result['n']:
            raise ValueError('Wrong sample count')
        keys = []
        for text in row['texts']:
            pair = (extract_answer(text), problem['gt'])
            key = hashlib.sha256(json.dumps(pair).encode()).hexdigest()
            pairs[key] = pair
            keys.append(key)
        row_keys.append(keys)
    missing = [(k, p) for k, p in pairs.items() if k not in cache['grades']]
    print(f'{source}: {len(rows)}/{len(problems)} problems; {len(missing)} uncached answer pairs', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(missing), 128):
            batch = missing[start:start+128]
            for (key, _), value in zip(batch, pool.map(grade_pair, [p for _, p in batch])):
                cache['grades'][key] = value
            common.save_json(cache_path, cache)
            print(f'Graded {min(start+128, len(missing))}/{len(missing)} new pairs', flush=True)
    flips = {'incorrect_to_correct': 0, 'correct_to_incorrect': 0}
    for row, keys in zip(rows, row_keys):
        old = row['correct']
        row['original_correct'] = old
        row['correct'] = [cache['grades'][k] for k in keys]
        flips['incorrect_to_correct'] += sum(a < b for a, b in zip(old, row['correct']))
        flips['correct_to_incorrect'] += sum(a > b for a, b in zip(old, row['correct']))
    result['original_metrics'] = result.get('metrics')
    result['benchmarks'] = list(sets)
    result['partial'] = len(rows) != len(problems)
    result['metrics'] = common.metrics_by_set(problems[:len(rows)], [int(sum(r['correct'])) for r in rows], result['n'])
    result['grading'] = {**identity, 'revision': 'required-author-backend-scientific-literals-v1',
                         'source': str(source.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                         'dataset_sha256': {s: hashlib.sha256((common.DATA/s/'test.parquet').read_bytes()).hexdigest() for s in sets},
                         'changes': flips}
    common.save_json(out, result)
    print(json.dumps({'out': str(out), 'changes': flips, 'metrics': result['metrics']['by_benchmark']}), flush=True)
    return result


def grade_pair(pair: tuple[str | None, str]) -> float:
    """Apply the shared grader to an already-extracted answer and its reference."""
    answer, ground_truth = pair
    return common.grade('' if answer is None else r'\boxed{'+answer+'}', ground_truth)


if __name__ == '__main__':
    main()
