#!/usr/bin/env python3
"""Run the original linear-head Golden v2 inference path from saved checkpoints."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biometaner.evaluation import TYPES, load_golden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--golden', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    gold = load_golden(args.golden)
    stage = args.experiment_dir / 'stage2'
    stage = stage if stage.is_dir() else args.experiment_dir
    checkpoints = {t: stage / t / 'best_model' for t in TYPES}
    if not all(p.is_dir() for p in checkpoints.values()):
        parser.error('Expected four linear BIO best_model directories, optionally under stage2/')
    import torch
    from biometaner.inference import predict_hf, merge_preds
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    predictions = {t: predict_hf(p, gold, t, device) for t, p in checkpoints.items()}
    merged = merge_preds(predictions, len(gold))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        for g, spans in zip(gold, merged):
            handle.write(json.dumps(dict(pmid=g['pmid'], text=g['text'], gold_entities=g['entities'],
                                         pred_entities=spans), ensure_ascii=False) + '\n')
    print(f'Wrote {len(gold)} abstracts; raw-logit span ranking, original windowing and type order')


if __name__ == '__main__':
    main()
