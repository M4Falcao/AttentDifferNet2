"""Regenerates the README figures from the reported numbers (no model or dataset needed).

    python assets/make_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['glass', 'lightning', 'polymer', 'vari', 'yoke']

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']                    # categorical slots 1-3
LADDER = ['#86b6ef', '#5598e7', '#256abf', '#0d366b']         # ordinal blue ramp

# Pixel AUROC on the holdout half of the test set (one head per class).
LADDER_ROWS = [
    ('Legacy (per-image min-max, σ=4)', [0.8864, 0.7658, 0.8534, 0.8594, 0.8373]),
    ('CFLOW-AD aggregation', [0.8761, 0.7940, 0.8461, 0.9051, 0.9115]),
    ('Best inference recipe, baseline head', [0.9156, 0.8674, 0.8653, 0.9220, 0.9449]),
    ('Best training variant + recipe', [0.9467, 0.8965, 0.8848, 0.9295, 0.9466]),
]

# Complete test set: image head (SEDifferNet) and pixel head with the `general` recipe.
IMAGE_AUROC = [0.8352, 0.9907, 0.9680, 0.9770, 0.9840]
PIXEL_AUROC = [0.9242, 0.8665, 0.8673, 0.9003, 0.9328]
AUPRO = [0.7670, 0.5711, 0.6221, 0.7365, 0.8182]
PIXEL_AUROC_NOBG = [0.6800, 0.9208, 0.6671, 0.8972, 0.9252]

# Paired effects on Pixel AUROC (validation half), mean over classes, 95% hierarchical bootstrap CI.
EFFECTS = [
    ('Gaussian σ=8 vs none', 0.0164, 0.0113, 0.0213),
    ('4 flips TTA vs none', 0.0048, 0.0015, 0.0083),
    ('Reflect padding 112 vs none', 0.0024, 0.0005, 0.0041),
    ('per_level_std vs CFLOW-AD agg.', 0.0017, -0.0046, 0.0081),
    ('raw NLL vs CFLOW-AD agg.', 0.0009, -0.0085, 0.0083),
    ('Levels L1+L3 vs all', 0.0003, -0.0142, 0.0177),
    ('Positional calibration', -0.0248, -0.0407, -0.0093),
    ('Levels L2+L3 vs all', -0.0505, -0.0720, -0.0302),
    ('Per-image min-max vs CFLOW-AD agg.', -0.0750, -0.1084, -0.0462),
]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=160, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def dot_rows(ax, labels, series, colors, names, xlim):
    """Cleveland dot plot: one row per label, one dot per series."""
    y = np.arange(len(labels))[::-1]
    for values, color, name in zip(series, colors, names):
        ax.scatter(values, y, s=60, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name)
    lo = np.min(series, axis=0)
    hi = np.max(series, axis=0)
    ax.hlines(y, lo, hi, color=GRID, linewidth=2, zorder=2)
    ax.set_yticks(y, labels, color=INK)
    ax.set_xlim(*xlim)
    style(ax)


def ladder_figure():
    labels = CLASSES + ['Mean']
    series = [vals + [float(np.mean(vals))] for _, vals in LADDER_ROWS]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    dot_rows(ax, labels, series, LADDER, [n for n, _ in LADDER_ROWS], (0.74, 0.97))
    ax.set_xlabel('Pixel AUROC (holdout)', color=MUTED)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, labelcolor=INK)
    ax.set_title('Pixel AUROC from the legacy pipeline to the best recipe', color=INK, loc='left')
    save(fig, 'pixel_auroc_ladder.png')


def full_test_figure():
    x = np.arange(len(CLASSES))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    for i, (values, name) in enumerate(zip((IMAGE_AUROC, PIXEL_AUROC, AUPRO),
                                           ('Image AUROC', 'Pixel AUROC', 'AUPRO'))):
        ax.bar(x + (i - 1) * width, values, width - 0.03, color=SERIES[i], label=name, zorder=3)
    ax.set_xticks(x, CLASSES, color=INK)
    ax.set_ylim(0, 1.05)
    style(ax)
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, labelcolor=INK)
    ax.set_title('Complete test set, per class', color=INK, loc='left')
    save(fig, 'per_class_metrics.png')


def effects_figure():
    names = [e[0] for e in EFFECTS][::-1]
    mean = np.array([e[1] for e in EFFECTS])[::-1]
    lo = np.array([e[2] for e in EFFECTS])[::-1]
    hi = np.array([e[3] for e in EFFECTS])[::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.hlines(y, lo, hi, color=SERIES[0], linewidth=2, zorder=2)
    ax.scatter(mean, y, s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(y, names, color=INK)
    style(ax)
    ax.set_xlabel('Δ Pixel AUROC (mean over classes, 95% CI)', color=MUTED)
    ax.set_title('Paired effect of each inference choice', color=INK, loc='left')
    save(fig, 'inference_effects.png')


def background_figure():
    fig, ax = plt.subplots(figsize=(8, 2.9), facecolor=SURFACE)
    dot_rows(ax, CLASSES, [PIXEL_AUROC, PIXEL_AUROC_NOBG], SERIES[:2],
             ['Original images', 'Background removed (retrained)'], (0.6, 1.0))
    ax.set_xlabel('Pixel AUROC (complete test set, general recipe)', color=MUTED)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, labelcolor=INK)
    ax.set_title('Effect of removing the background', color=INK, loc='left')
    save(fig, 'background_removal.png')


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})
    ladder_figure()
    full_test_figure()
    effects_figure()
    background_figure()
    print('figures written to', HERE)
