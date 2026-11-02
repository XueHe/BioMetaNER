#!/usr/bin/env python3
"""Four shared-scale per-type panels from count-derived seed summaries."""

import argparse
import csv
from pathlib import Path

BACKBONES = ['BERT-base', 'BioBERT', 'SciBERT', 'PubMedBERT', 'BioMegatron']
TYPES = ['MIC', 'DIS', 'CHE', 'HAB']
EXPERIMENTS = [f'E{i}' for i in range(1, 10)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                                'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    with args.summary.open() as handle:
        rows = list(csv.DictReader(handle))
    data = {(r['type'], r['backbone'], r['experiment']): r for r in rows
            if not r.get('match') or r['match'] == 'partial'}
    fig = plt.figure(figsize=(6, 4.8))
    gs = fig.add_gridspec(2, 2, left=.14, right=.98, top=.88, bottom=.18, wspace=.06, hspace=.5)
    cmap, norm = matplotlib.colormaps['cividis'], Normalize(0, 100)
    for index, typ in enumerate(TYPES):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        values = np.full((5, 9), np.nan)
        for y, backbone in enumerate(BACKBONES):
            for x, exp in enumerate(EXPERIMENTS):
                r = data.get((typ, backbone, exp))
                if r and r['mean']:
                    values[y, x] = 100 * float(r['mean'])
        ax.pcolormesh(np.arange(10) - .5, np.arange(6) - .5,
                      np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                      edgecolors='white', linewidth=.25, rasterized=False)
        ax.set_xlim(-.5, 8.5)
        ax.set_ylim(4.5, -.5)
        ax.set_facecolor('#e8e8e8')
        ax.set_xticks(range(9), EXPERIMENTS)
        ax.xaxis.tick_top()
        ax.set_yticks(range(5), BACKBONES if index % 2 == 0 else [''] * 5)
        ax.tick_params(length=0, pad=4)
        ax.set_title(typ, loc='left', pad=24, fontweight='bold')
        for label in ax.get_xticklabels():
            if label.get_text() in ['E6', 'E9']:
                label.set_fontweight('bold')
                label.set_bbox(dict(facecolor='white', edgecolor='black', pad=1))
        for y, backbone in enumerate(BACKBONES):
            for x, exp in enumerate(EXPERIMENTS):
                r = data.get((typ, backbone, exp))
                value = values[y, x]
                if np.isnan(value):
                    ax.text(x, y, '--', ha='center', va='center', fontsize=6)
                    continue
                rgb = cmap(norm(value))[:3]
                linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
                luminance = sum(v * w for v, w in zip(linear, [.2126, .7152, .0722]))
                color = 'black' if (luminance + .05) / .05 >= 1.05 / (luminance + .05) else 'white'
                sd = f'{100 * float(r["sample_sd"]):.2f}' if r['sample_sd'] else '--'
                ax.text(x, y, f'{value:.2f}\n{sd}', ha='center', va='center', fontsize=6, color=color, linespacing=1.35)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.14, .09, .84, .018])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    bar.set_label('Partial-match F1 (%)')
    bar.solids.set_rasterized(False)
    fig.text(.14, .135, 'Cells: mean (top), sample SD (bottom); see CSV for seed coverage.', fontsize=6)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(args.output / f'figure4.{extension}', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
