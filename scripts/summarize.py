#!/usr/bin/env python3
"""Recompute per-type and Macro seed statistics from integer count CSV."""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biometaner.statistics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.counts.open() as handle:
        rows = summarize(list(csv.DictReader(handle)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['match', 'experiment', 'backbone', 'type', 'mean', 'sample_sd', 'n_seeds', 'seeds'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} count-derived summaries to {args.output}')


if __name__ == '__main__':
    main()
