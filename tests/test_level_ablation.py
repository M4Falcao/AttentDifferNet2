"""Level ablation on a synthetic NLL cache."""

import numpy as np

import config as c
from core.level_ablation import level_subsets, run_level_ablation

RECIPE = dict(c.PIXEL_INFERENCE_DEFAULTS, score_mode='raw', sigma=0.0, levels='012')
OUT = c.pixel_out_size


def _case(informative_level, noise_scale, n=8, seed=0):
    """Only ``informative_level`` separates the defect; the other levels carry noise of scale ``noise_scale``."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)], dtype=np.uint8)
    nll = rng.normal(size=(n, 3, OUT, OUT)).astype(np.float32) * noise_scale
    masks = []
    for i, label in enumerate(labels):
        if not label:
            masks.append(None)
            continue
        cell = np.zeros((OUT, OUT), dtype=bool)
        cell[30:50, 40:60] = True
        nll[i, informative_level][cell] += 6.0
        rows = np.arange(c.img_size[0]) * OUT // c.img_size[0]
        cols = np.arange(c.img_size[1]) * OUT // c.img_size[1]
        masks.append(cell[np.ix_(rows, cols)])
    return nll, masks, labels


def test_subsets_of_three_levels():
    assert level_subsets([0, 1, 2]) == ['0', '1', '2', '01', '02', '12', '012']
    assert level_subsets([0, 2]) == ['0', '2', '02']


def test_selects_single_informative_level_when_others_are_noise():
    nll, masks, labels = _case(informative_level=0, noise_scale=3.0)
    summary = run_level_ablation(nll, masks, labels, RECIPE, [0, 1, 2], None, min_gain=0.005)
    assert summary['status'] == 'done'
    assert summary['selected'] == '0'
    assert len(summary['table']) == 7 and summary['selected_on'] == 'test'


def test_keeps_all_levels_when_gain_is_below_threshold():
    nll, masks, labels = _case(informative_level=0, noise_scale=3.0)
    summary = run_level_ablation(nll, masks, labels, RECIPE, [0, 1, 2], None, min_gain=1.0)
    assert summary['selected'] == '012'


def test_skips_without_ground_truth():
    nll, _, labels = _case(informative_level=0, noise_scale=1.0)
    summary = run_level_ablation(nll, [None] * len(labels), labels, RECIPE, [0, 1, 2], None, 0.005)
    assert summary['status'].startswith('skipped') and summary['selected'] == '012'
