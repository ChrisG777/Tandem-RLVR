"""Reconstruct Figure 3's marginal token diagnostics from saved Figure 2 solo texts.

See docs/FIGURE3.md for paper definitions and explicit reconstruction choices.
No inference or answer grading is performed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import rel_entr
from scipy.stats import spearmanr
from tokenizers import Tokenizer

import config
from figure2 import EXPECTED_SIZES

ARMS = ('base', 'grpo', 'tandem')
COLORS = {'base': '#969696', 'grpo': '#298eb0', 'tandem': '#ad357a'}
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
# Manual interpretability exclusions from the ranked candidates; keep IDs auditable.
CASE_STUDY_FRAGMENTS = {62944, 3614, 17272, 13842, 1211, 763}


def main() -> None:
    """Read three complete solo JSONs; write PNG/SVG, statistics and cached counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='output filename stem')
    parser.add_argument('--tokenizer', type=Path, required=True, help='pinned tokenizer.json')
    parser.add_argument('--pseudocount', type=float, default=0.5)
    parser.add_argument('--top', type=int, default=500)
    parser.add_argument('--min-base-frequency', type=float, default=1e-5)
    args = parser.parse_args()
    if args.pseudocount <= 0 or args.top < 2 or not 0 < args.min_base_frequency < 1:
        parser.error('require positive pseudocount, top >= 2, and 0 < minimum frequency < 1')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    tokenizer_hash = _sha256(args.tokenizer)
    counts, sources, problem_keys = {}, {}, None
    for arm in ARMS:
        path = args.results / arm / 'solo.json'
        source_hash = _sha256(path)
        cache = args.out.parent / f'figure3-counts-{arm}.json'
        identity = {'source_sha256': source_hash, 'tokenizer_sha256': tokenizer_hash,
                    'method': 'saved-text-encode-no-specials-v1'}
        saved = json.loads(cache.read_text()) if cache.exists() else {}
        if saved.get('identity') == identity:
            print(f'[{arm}] reuse verified token counts', flush=True)
        else:
            result = json.loads(path.read_text())
            keys = _validate(result)
            counted = count_tokens(result, tokenizer)
            saved = {'identity': identity, 'keys': sorted(keys), 'model': result['model'],
                     'n': result['n'], 'counts': counted.tolist()}
            cache.write_text(json.dumps(saved) + '\n')
        keys = {tuple(k) for k in saved['keys']}
        if problem_keys is not None and keys != problem_keys:
            raise ValueError('The three arms do not contain identical problems')
        problem_keys = keys
        counts[arm] = np.array(saved['counts'], dtype=np.int64)
        sources[arm] = {'path': str(path.resolve()), **identity, 'model': saved['model'],
                        'problems': len(keys), 'samples_per_problem': saved['n'],
                        'tokens': int(counts[arm].sum())}
    if len({s['samples_per_problem'] for s in sources.values()}) != 1:
        raise ValueError('The three arms have different numbers of samples per problem')
    stats = summarize(counts, tokenizer, pseudocount=args.pseudocount,
                      top=args.top, min_base_frequency=args.min_base_frequency)
    stats['sources'] = sources
    stats['tokenizer'] = {'path': str(args.tokenizer.resolve()), 'sha256': tokenizer_hash,
                          'expected_base_revision': REVISION}
    args.out.with_suffix('.json').write_text(json.dumps(stats, indent=2, allow_nan=False) + '\n')
    plot(stats, args.out)
    print(json.dumps({k: stats[k] for k in ('jsd', 'jsd_change_percent', 'scatter_summary')}, indent=2))


def count_tokens(result: dict, tokenizer: Tokenizer) -> np.ndarray:
    """Pool every saved solo response, including final answers, without special tokens.

    Return integer counts indexed by token ID. No filtering by correctness; prompts
    are absent from the saved texts. This re-encodes text, not original sampler IDs.
    """
    counts = np.zeros(tokenizer.get_vocab_size(), dtype=np.int64)
    for start in range(0, len(result['gens']), 32):
        texts = [text for row in result['gens'][start:start + 32] for text in row['texts']]
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        ids = np.concatenate([np.asarray(e.ids, dtype=np.int64) for e in encodings])
        counts += np.bincount(ids, minlength=len(counts))
        print(f"[count] {min(start + 32, len(result['gens']))}/{len(result['gens'])} problems", flush=True)
    return counts


def summarize(counts: dict[str, np.ndarray], tokenizer: Tokenizer, *,
              pseudocount: float = 0.5, top: int = 500,
              min_base_frequency: float = 1e-5) -> dict:
    """Compute pooled marginal drift in natural-log units; return JSON-ready panels.

    JSD uses empirical probabilities with exact zero handling. Log ratios use an
    additive pseudocount on the observed union. Survival is senior-token weighted;
    a uniform token-type alternative and cutoff sensitivities are also retained.
    """
    if pseudocount <= 0 or top < 2 or not 0 < min_base_frequency < 1:
        raise ValueError('Invalid smoothing or selection parameters')
    arrays = np.stack([counts[a] for a in ARMS])
    if (arrays < 0).any() or (arrays.sum(axis=1) <= 0).any():
        raise ValueError('Counts must be nonnegative with nonempty arms')
    ids = np.flatnonzero(arrays.sum(axis=0))
    arrays = arrays[:, ids]
    p = arrays / arrays.sum(axis=1, keepdims=True)
    smooth = (arrays + pseudocount) / (arrays.sum(axis=1, keepdims=True) + pseudocount * len(ids))
    drift = np.log(smooth[1:] / smooth[0])
    bins = np.searchsorted([1e-5, 1e-4, 1e-3], p[0], side='right')
    grid = np.linspace(0, max(2.25, float(np.quantile(np.abs(drift), .99))), 301)
    jsd, contributions, survival, uniform = {}, {}, {}, {}
    for i, arm in enumerate(ARMS[1:], 1):
        midpoint = (p[0] + p[i]) / 2
        term = .5 * (rel_entr(p[0], midpoint) + rel_entr(p[i], midpoint))
        contributions[arm] = [float(term[bins == b].sum()) for b in range(4)]
        jsd[arm] = float(term.sum())
        survival[arm] = [float(p[i, np.abs(drift[i-1]) >= x].sum()) for x in grid]
        uniform[arm] = [float(np.mean(np.abs(drift[i-1]) >= x)) for x in grid]
    eligible = np.flatnonzero(p[0] >= min_base_frequency)
    ranked = eligible[np.argsort(-np.abs(drift[0, eligible]), kind='stable')]
    selected = ranked[:top]
    x, y = drift[0, selected], drift[1, selected] - drift[0, selected]
    scatter_summary = _scatter_summary(x, y)
    cases = []
    for i, arm in ((2, 'tandem'), (1, 'grpo')):
        candidates = eligible[np.argsort(-drift[i-1, eligible], kind='stable')]
        used = set()
        for j in candidates:
            text = tokenizer.decode([int(ids[j])], skip_special_tokens=False)
            label = text.strip()
            # Fixed, auditable filter, independent of a desired scientific outcome.
            readable = bool(re.fullmatch(r"[A-Za-z][A-Za-z '’-]{1,23}|[#:*_=-]{2,12}", label))
            if not readable or int(ids[j]) in CASE_STUDY_FRAGMENTS or label in used or p[i, j] <= max(p[0, j], p[3-i, j]):
                continue
            cases.append({'arm': arm, 'id': int(ids[j]), 'token_text': text,
                          'label': label, 'rates_per_million': dict(zip(ARMS, (p[:, j]*1e6).tolist())),
                          'ratio_to_base': float(p[i, j]/p[0, j])})
            used.add(label)
            if len(used) == 6:
                break
    return {
        'protocol': 'Figure 3 reconstruction; choices not specified by paper are explicit in docs/FIGURE3.md',
        'settings': {'pseudocount': pseudocount, 'top': top, 'min_base_frequency': min_base_frequency,
                     'pooling': 'all response tokens, all problems and samples; no macro weighting',
                     'survival_weighting': 'senior empirical token mass', 'log_base': 'e',
                     'jsd_smoothing': 'none; zero terms handled exactly', 'support_size': len(ids)},
        'jsd': jsd,
        'jsd_change_percent': 100*(jsd['tandem']/jsd['grpo']-1) if jsd['grpo'] else None,
        'jsd_by_base_frequency': contributions,
        'survival': {'threshold': grid.tolist(), **survival},
        'uniform_token_type_survival': {'threshold': grid.tolist(), **uniform},
        'scatter': {'token_ids': ids[selected].tolist(), 'x': x.tolist(), 'y': y.tolist(),
                    'closer_to_base': (np.abs(x+y) < np.abs(x)).tolist()},
        'scatter_summary': scatter_summary,
        'top_cutoff_sensitivity': {str(k): _scatter_summary(drift[0, ranked[:k]],
                                  drift[1, ranked[:k]]-drift[0, ranked[:k]]) for k in (100, 250, 500, 1000)},
        'case_study': cases,
        'case_study_excluded_fragment_ids': sorted(CASE_STUDY_FRAGMENTS),
    }


def plot(stats: dict, out: Path) -> None:
    """Write the four-panel PNG/SVG; visibly label reconstruction choices."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(12.5, 8.6), layout='constrained')
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    a, b, c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    d = fig.add_subplot(gs[1, :])
    for arm in ARMS[1:]:
        a.plot(stats['survival']['threshold'], stats['survival'][arm], color=COLORS[arm], label=arm.upper() if arm == 'grpo' else 'Tandem', lw=2)
    a.set(yscale='log', ylim=(1e-5, 1.05), xlabel='Absolute log ratio to base (nats)', ylabel='Fraction of emitted tokens', title='a  Token-frequency drift')
    a.legend(frameon=False)
    a.text(.03, .04, 'Weighted by each senior’s token counts', transform=a.transAxes, fontsize=8)
    for i, arm in enumerate(ARMS[1:]):
        b.bar(np.arange(4)+(i-.5)*.36, stats['jsd_by_base_frequency'][arm], .36, color=COLORS[arm])
    for i, (g, t) in enumerate(zip(stats['jsd_by_base_frequency']['grpo'], stats['jsd_by_base_frequency']['tandem'])):
        if g:
            b.annotate(f'{100*(t/g-1):+.0f}%', (i, max(g,t)), xytext=(0,5), textcoords='offset points', ha='center', fontsize=9)
    b.set_xticks(range(4), ['<10⁻⁵', '10⁻⁵–10⁻⁴', '10⁻⁴–10⁻³', '≥10⁻³'], rotation=15)
    b.set(xlabel='Token frequency in base', ylabel='JSD contribution (nats)', title='b  Divergence by frequency band')
    b.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    b.margins(y=.25)
    x, y = np.array(stats['scatter']['x']), np.array(stats['scatter']['y'])
    closer = np.array(stats['scatter']['closer_to_base'])
    c.scatter(x[closer], y[closer], s=9, alpha=.5, color=COLORS['grpo'])
    c.scatter(x[~closer], y[~closer], s=9, alpha=.6, color='#e8a032')
    bound = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.1
    c.plot([-bound,bound],[bound,-bound], '--', color='.6', lw=1)
    c.axhline(0, color='.8', lw=.7)
    c.axvline(0, color='.8', lw=.7)
    c.set(xlim=(-bound,bound), ylim=(-bound,bound), xlabel='Base → GRPO: log(pGRPO / pbase)', ylabel='GRPO → Tandem: log(pTandem / pGRPO)', title='c  Does Tandem undo GRPO’s drift?')
    summary = stats['scatter_summary']
    c.text(.03, .97, f"{summary['n']} tokens; {summary['closer_percent']:.1f}% closer\nSpearman ρ = {summary['spearman_rho']:.2f}", transform=c.transAxes, va='top', fontsize=9)
    cases = stats['case_study']
    for row, item in enumerate(cases):
        arm = item['arm']; other = 'tandem' if arm == 'grpo' else 'grpo'
        rates = item['rates_per_million']; color = COLORS[arm]
        d.plot([rates['base'],rates[arm]], [row,row], color='.85', lw=2)
        d.scatter(rates['base'],row, c=COLORS['base'], s=24, zorder=3)
        d.scatter(rates[arm],row, c=color, s=32, zorder=4)
        d.scatter(rates[other],row, facecolors='none', edgecolors=COLORS[other], s=26, zorder=4)
        d.annotate(f"{item['ratio_to_base']:.2f}×", (rates[arm],row), xytext=(9,0), textcoords='offset points', va='center', fontsize=9, color=color)
    d.set_yticks(range(len(cases)), [repr(r['token_text']) for r in cases], fontfamily='monospace')
    d.invert_yaxis()
    d.set(xscale='log', xlabel='Emission rate per million tokens', title='d  Over-emitted tokens (ranked by senior / base rate)')
    split = sum(r['arm'] == 'tandem' for r in cases)
    d.axhline(split-.5, color='.85', lw=.7)
    for arm, yloc in [('tandem',.77),('grpo',.24)]:
        d.text(.99, yloc, ('Tandem' if arm=='tandem' else 'GRPO')+' over-emits', color=COLORS[arm], transform=d.transAxes, ha='right', weight='bold')
    d.margins(x=.2, y=.12)
    if cases:
        d.set_xlim(right=4 * max(max(item['rates_per_million'].values()) for item in cases))
    d.legend(handles=[Line2D([],[],marker='o',ls='',color=COLORS[a],label=('Base (junior)' if a=='base' else a.upper() if a=='grpo' else 'Tandem')) for a in ARMS] + [Line2D([],[],marker='o',ls='',markerfacecolor='none',color='.4',label='Other senior (open)')], loc='upper left', bbox_to_anchor=(0,1.02), ncol=4, frameon=False, fontsize=9)
    for ax in (a,b,c,d):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Figure 3 reconstruction · same saved solo evaluations as Figure 2', fontsize=15, weight='bold')
    fig.supxlabel('1,064 problems × 32 responses per model · pooled response tokens · explicit reconstruction conventions in docs/FIGURE3.md', fontsize=9)
    fig.savefig(out.with_suffix('.png'), dpi=200)
    fig.savefig(out.with_suffix('.svg'))
    plt.close(fig)


def _validate(result: dict) -> set[tuple[str, int]]:
    if result['phase'] != 'solo' or result.get('limit',0) or result.get('partial') or result['n'] < 32 or result['sampling'] != config.record():
        raise ValueError('Require a complete Figure 2 solo evaluation with original decoding')
    keys = set()
    sizes = Counter()
    for row in result['gens']:
        key = (row['set'], row['idx'])
        if key in keys or len(row['texts']) != result['n'] or not all(isinstance(t,str) for t in row['texts']):
            raise ValueError(f'Duplicate problem or invalid response count: {key}')
        keys.add(key)
        sizes['aime24_26' if row['set'] in ('aime24','aime25','aime26') else row['set']] += 1
    if dict(sizes) != EXPECTED_SIZES:
        raise ValueError('Expected all 1,064 Figure 2 problems')
    return keys


def _scatter_summary(x: np.ndarray, y: np.ndarray) -> dict:
    rho = float(spearmanr(x,y).statistic) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None
    return {'n': len(x), 'closer_percent': float(100*np.mean(np.abs(x+y)<np.abs(x))) if len(x) else None,
            'opposite_direction_percent': float(100*np.mean(x*y<0)) if len(x) else None,
            'spearman_rho': rho}


def _sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if __name__ == '__main__':
    main()
