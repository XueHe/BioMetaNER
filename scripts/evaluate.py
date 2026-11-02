#!/usr/bin/env python3
"""Evaluate saved character-span predictions; no model loading."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biometaner.evaluation import load_golden, read_jsonl, score, sha256, ORIGINAL_MATCHER_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--golden', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-unpinned', action='store_true', help='Synthetic/other corpora only, not Golden v2 reporting')
    args = parser.parse_args()
    result = score(load_golden(args.golden, pinned=not args.allow_unpinned), read_jsonl(args.predictions))
    result['provenance'] = dict(golden_sha256=sha256(args.golden),
                                predictions_sha256=sha256(args.predictions),
                                original_matcher_sha256=ORIGINAL_MATCHER_SHA256,
                                golden_v2_pinned=not args.allow_unpinned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for mode in ('partial', 'strict'):
        print(f'{mode}: Macro F1 {100 * result[mode]["macro_F1"]:.2f}%; micro F1 {100 * result[mode]["micro_F1"]:.2f}%')


if __name__ == '__main__':
    main()
