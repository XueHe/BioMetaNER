#!/usr/bin/env python3
"""Paired whole-abstract bootstrap from text-free integer counts."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biometaner.evaluation import GOLDEN_SHA256, GOLDEN_COUNTS, ORIGINAL_MATCHER_SHA256

PAIRS = [('E6', 'E2'), ('E6', 'E4'), ('E6', 'E8'), ('E6', 'E9'),
         ('E4', 'E8'), ('E4', 'E9'), ('E9', 'E8')]
TYPES = ['MIC', 'DIS', 'CHE', 'HAB', 'Macro']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.n_resamples < 2:
        parser.error('At least two bootstrap draws are required')
    import numpy as np
    with gzip.open(args.counts, 'rt') as handle:
        payload = json.load(handle)
    if payload['golden_sha256'] != GOLDEN_SHA256 or payload['matcher_sha256'] != ORIGINAL_MATCHER_SHA256:
        raise ValueError('Golden or matcher version mismatch')
    runs = payload['runs']
    n = len(payload['documents'])
    if n != 41 or len({(r['experiment'], r['backbone'], r['seed']) for r in runs}) != len(runs):
        raise ValueError('Unexpected document dimension or duplicate run')
    rng = random.Random(args.seed)
    indices = np.array([[rng.randrange(n) for _ in range(n)] for _ in range(args.n_resamples)], dtype=np.int16)
    weights = np.array([np.bincount(row, minlength=n) for row in indices], dtype=np.int64)
    groups = list(dict.fromkeys((r['experiment'], r['backbone']) for r in runs))
    lookup = {(r['experiment'], r['backbone'], r['seed']): i for i, r in enumerate(runs)}
    summaries, contrasts = [], []

    def ratios(counts):
        numerator = 2 * counts[..., 0]
        denominator = numerator + counts[..., 1] + counts[..., 2]
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)

    def interval(values, point):
        low, high = np.percentile(values, [2.5, 97.5], method='linear')
        return dict(point_estimate=float(point), ci95_low=float(low), ci95_high=float(high),
                    bootstrap_mean=float(np.mean(values)))

    for mode in ['partial', 'strict']:
        counts = np.array([r['counts'][mode] for r in runs], dtype=np.int64)
        if counts.shape != (len(runs), n, 4, 3) or (counts < 0).any():
            raise ValueError('Expected nonnegative run/document/type/TP-FP-FN counts')
        if not ((counts[..., 0] + counts[..., 2]).sum(axis=1) ==
                np.array([GOLDEN_COUNTS[t] for t in TYPES[:-1]])).all():
            raise ValueError('Per-run gold entity totals differ from Golden v2')
        point = ratios(counts.sum(axis=1))
        sampled = ratios(np.einsum('bd,rdtc->brtc', weights, counts, optimize=True))
        point = np.concatenate([point, point.mean(axis=-1, keepdims=True)], axis=-1)
        sampled = np.concatenate([sampled, sampled.mean(axis=-1, keepdims=True)], axis=-1)
        for exp, backbone in groups:
            ids = [i for i, r in enumerate(runs) if (r['experiment'], r['backbone']) == (exp, backbone)]
            for t, typ in enumerate(TYPES):
                summaries.append(dict(match=mode, experiment=exp, backbone=backbone, type=typ,
                                      n_seeds=len(ids), **interval(sampled[:, ids, t].mean(axis=1), point[ids, t].mean())))
        for backbone in dict.fromkeys(r['backbone'] for r in runs):
            for first, second in PAIRS:
                a = {r['seed'] for r in runs if (r['experiment'], r['backbone']) == (first, backbone)}
                b = {r['seed'] for r in runs if (r['experiment'], r['backbone']) == (second, backbone)}
                seeds = sorted(a & b)
                if not seeds:
                    continue
                left = [lookup[(first, backbone, s)] for s in seeds]
                right = [lookup[(second, backbone, s)] for s in seeds]
                for t, typ in enumerate(TYPES):
                    values = (sampled[:, left, t] - sampled[:, right, t]).mean(axis=1)
                    contrasts.append(dict(match=mode, backbone=backbone, comparison=f'{first}-{second}',
                                          type=typ, n_paired_seeds=len(seeds),
                                          **interval(values, (point[left, t] - point[right, t]).mean())))
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('summary_ci.csv', summaries), ('paired_differences_ci.csv', contrasts)]:
        with (args.output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        n_bootstrap=args.n_resamples, bootstrap_seed=args.seed, numpy=np.__version__,
        statistic='Per-seed type F1 and four-type macro, then fixed-seed mean',
        confidence_interval='95% percentile with linear interpolation',
        pairing='Same abstract draws and matched seed IDs',
        warning='No training-seed resampling or multiple-comparison correction'), indent=2) + '\n')
    print(f'Wrote {len(summaries)} estimates and {len(contrasts)} paired contrasts')


if __name__ == '__main__':
    main()
