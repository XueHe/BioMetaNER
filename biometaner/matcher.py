"""Original Golden v2 matcher functions; prediction order is preserved."""

def overlaps(a, b):
    return int(a["start"]) < int(b["end"]) and int(a["end"]) > int(b["start"])


def partial_counts(gold, pred, entity_type=None):
    if entity_type:
        gold = [g for g in gold if g.get("type") == entity_type]
        pred = [p for p in pred if p.get("type") == entity_type]
    matched = set()
    tp = 0
    for p in pred:
        for idx, g in enumerate(gold):
            if idx in matched: continue
            if p.get("type") != g.get("type"): continue
            if overlaps(p, g):
                matched.add(idx); tp += 1; break
    return tp, len(pred) - tp, len(gold) - tp
