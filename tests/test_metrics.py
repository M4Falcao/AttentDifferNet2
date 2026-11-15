"""Metrics accumulated image by image against exact references."""

import numpy as np
from sklearn.metrics import auc, average_precision_score, roc_auc_score

from core.metrics import PixelMetrics, ScoreRange, aupro_regions


def _synthetic(n=6, size=64, seed=0):
    rng = np.random.default_rng(seed)
    maps, masks = [], []
    for i in range(n):
        mask = np.zeros((size, size), dtype=bool)
        if i % 2:
            r, col = rng.integers(5, size - 20, 2)
            mask[r:r + 12, col:col + 9] = True
            mask[r + 4:r + 6, col + 3:col + 5] = False   # hole: exercises the hole filling
        amap = rng.normal(size=(size, size)) + 2.5 * mask
        maps.append(amap)
        masks.append(mask)
    return maps, masks


def _accumulate(maps, masks, **kwargs):
    rng_ = ScoreRange()
    for m in maps:
        rng_.add(m)
    metrics = PixelMetrics(rng_.lo, rng_.hi, **kwargs)
    for m, k in zip(maps, masks):
        metrics.add(m, k)
    return metrics.result()


def _aupro_reference(maps, masks, max_step=1000, expect_fpr=0.3):
    """Literal AUPRO loop from the official CFLOW-AD repository."""
    maps, masks = np.stack(maps), np.stack(masks)
    max_th, min_th = maps.max(), maps.min()
    delta = (max_th - min_th) / max_step
    pros_mean, fprs = [], []
    regions = [aupro_regions(m) for m in masks]
    for step in range(max_step):
        th = max_th - step * delta
        binary = maps > th
        pros = [binary[i][rows, cols].sum() / area
                for i, regs in enumerate(regions) for rows, cols, area in regs]
        inverse = ~masks
        fprs.append(np.logical_and(inverse, binary).sum() / inverse.sum())
        pros_mean.append(np.mean(pros))
    fprs, pros_mean = np.array(fprs), np.array(pros_mean)
    keep = fprs <= expect_fpr
    fprs, pros_mean = fprs[keep], pros_mean[keep]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    return auc(fprs, pros_mean)


def test_pixel_auroc_and_ap_match_sklearn():
    maps, masks = _synthetic()
    result = _accumulate(maps, masks, aupro=False)
    labels = np.concatenate([m.ravel() for m in masks])
    scores = np.concatenate([m.ravel() for m in maps])
    assert abs(result['pixel_auroc'] - roc_auc_score(labels, scores)) < 1e-4
    assert abs(result['pixel_ap'] - average_precision_score(labels, scores)) < 1e-3


def test_aupro_matches_official_loop():
    maps, masks = _synthetic()
    result = _accumulate(maps, masks, aupro=True)
    assert abs(result['aupro'] - _aupro_reference(maps, masks)) < 1e-3


def test_undefined_metrics_are_nan():
    maps = [np.random.default_rng(1).normal(size=(16, 16))]
    result = _accumulate(maps, [np.zeros((16, 16), dtype=bool)])
    assert all(np.isnan(v) for v in result.values())
