"""Pinned Golden v2 evaluation without model or scientific-library imports."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from .matcher import partial_counts

TYPES = ('MIC', 'DIS', 'CHE', 'HAB')
GOLDEN_SHA256 = 'e874a9d0148cf8dd460e7ac56ee36bcf2267792d70c6852fb3b9d20bbf1cde63'
GOLDEN_COUNTS = {'MIC': 497, 'DIS': 606, 'CHE': 420, 'HAB': 202}
ORIGINAL_MATCHER_SHA256 = '8575708511494ca7653fc0cc78e255c7ef9e4ab4f49ce1e030d43733697625bb'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_spans(spans, text):
    for span in spans:
        if span.get('type') not in TYPES:
            raise ValueError(f'Unknown entity type: {span.get("type")}')
        if any(type(span.get(k)) is not int for k in ('start', 'end')):
            raise ValueError('Character offsets must be integers')
        if not 0 <= span['start'] < span['end'] <= len(text):
            raise ValueError('Character span outside document')


def load_golden(path, pinned=True):
    digest = sha256(path)
    if pinned and digest != GOLDEN_SHA256:
        raise ValueError('Golden SHA-256 mismatch; refuse to mix Golden versions')
    rows = read_jsonl(path)
    ids = [str(row['pmid']) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate Golden document ID')
    for row in rows:
        validate_spans(row['entities'], row['text'])
    counts = Counter(e['type'] for row in rows for e in row['entities'])
    if pinned and (len(rows) != 41 or counts != GOLDEN_COUNTS):
        raise ValueError('Golden dimensions differ from 41 abstracts / 1725 entities')
    return rows


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def strict_counts(gold, pred, entity_type):
    def spans(values):
        return Counter((e['start'], e['end']) for e in values if e['type'] == entity_type)
    g, p = spans(gold), spans(pred)
    tp = sum((g & p).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def score(gold, predictions):
    by_id = {str(row['pmid']): row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {str(g['pmid']) for g in gold}:
        raise ValueError('Prediction IDs must cover each Golden abstract exactly once')
    totals = {mode: {t: [0, 0, 0] for t in TYPES} for mode in ('partial', 'strict')}
    document_counts = []
    for g in gold:
        p = by_id[str(g['pmid'])]
        if p['text'] != g['text']:
            raise ValueError(f'Text mismatch for document {g["pmid"]}')
        validate_spans(p['pred_entities'], g['text'])
        if 'gold_entities' in p:
            key = lambda e: (e['start'], e['end'], e['type'])
            if Counter(map(key, p['gold_entities'])) != Counter(map(key, g['entities'])):
                raise ValueError('Saved gold entities differ from supplied Golden')
        doc = {'document_index': len(document_counts)}
        for mode, matcher in (('partial', partial_counts), ('strict', strict_counts)):
            doc[mode] = []
            for t in TYPES:
                values = matcher(g['entities'], p['pred_entities'], t)
                doc[mode].append(list(values))
                totals[mode][t] = [a + b for a, b in zip(totals[mode][t], values)]
        document_counts.append(doc)
    result = {}
    for mode, values in totals.items():
        result[mode] = {}
        for t, (tp, fp, fn) in values.items():
            result[mode][t] = dict(TP=tp, FP=fp, FN=fn,
                                   P=tp / (tp + fp) if tp + fp else 0.0,
                                   R=tp / (tp + fn) if tp + fn else 0.0,
                                   F1=f1(tp, fp, fn))
        result[mode]['macro_F1'] = statistics.mean(result[mode][t]['F1'] for t in TYPES)
        pooled = [sum(values[t][i] for t in TYPES) for i in range(3)]
        result[mode]['micro_F1'] = f1(*pooled)
    result['document_counts'] = document_counts
    return result
