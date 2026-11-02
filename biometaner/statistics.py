"""Aggregate integer-count F1 within seed before reporting mean/sample SD."""

from collections import defaultdict
import statistics

from .evaluation import TYPES, GOLDEN_COUNTS, f1


def summarize(rows, check_golden=True):
    by_run = defaultdict(dict)
    for row in rows:
        mode = row.get('match') or 'partial'
        key = (mode, row['experiment'], row['backbone'], int(row['seed']))
        entity_type = row['type']
        if entity_type not in TYPES or entity_type in by_run[key]:
            raise ValueError('Unknown type or duplicate run/seed/type row')
        counts = tuple(int(row[k]) for k in ('TP', 'FP', 'FN'))
        if min(counts) < 0:
            raise ValueError('Negative count')
        if check_golden and counts[0] + counts[2] != GOLDEN_COUNTS[entity_type]:
            raise ValueError('Gold count differs from pinned Golden v2')
        by_run[key][entity_type] = f1(*counts)
    groups = defaultdict(list)
    for (mode, exp, backbone, seed), values in by_run.items():
        for t, value in values.items():
            groups[(mode, exp, backbone, t)].append((seed, value))
        if set(values) == set(TYPES):
            groups[(mode, exp, backbone, 'Macro')].append((seed, statistics.mean(values.values())))
    output = []
    for (mode, exp, backbone, t), entries in groups.items():
        values = [value for _, value in entries]
        output.append(dict(match=mode, experiment=exp, backbone=backbone, type=t,
                           mean=statistics.mean(values),
                           sample_sd=statistics.stdev(values) if len(values) > 1 else '',
                           n_seeds=len(values), seeds=';'.join(str(seed) for seed, _ in entries)))
    return output
