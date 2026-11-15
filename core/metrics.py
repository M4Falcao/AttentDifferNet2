"""Image-level and pixel-level metrics.

Pixel metrics are accumulated image by image so that all test maps never need to
be held in memory (yoke has 1253 images x 448**2 pixels). This takes two passes:
the first finds the score range (:class:`ScoreRange`) and the second accumulates
(:class:`PixelMetrics`).

- **Pixel AUROC / AP**: histogram with ``2**20`` bins over the score range. Only
  pixels falling into the same bin become ties; the difference to the exact
  scikit-learn computation stays below 1e-4 (see ``tests/test_metrics.py``).
- **AUPRO**: implementation of the official CFLOW-AD repository — 1000 global
  thresholds, FPR over the normal pixels of all images, mean overlap per region,
  FPR <= 0.3 rescaled to [0, 1] before the area.
"""

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as label_regions, regionprops
from sklearn.metrics import auc, average_precision_score, roc_auc_score


class ScoreRange:
    """Minimum and maximum of a sequence of maps (first pass)."""

    def __init__(self):
        self.lo, self.hi = np.inf, -np.inf

    def add(self, amap):
        self.lo = min(self.lo, float(np.min(amap)))
        self.hi = max(self.hi, float(np.max(amap)))


class PixelMetrics:
    """Pixel AUROC, pixel AP and AUPRO accumulated image by image (second pass).

    Args:
        lo, hi: Score range, from :class:`ScoreRange`.
        aupro: Also compute AUPRO (costs one region labelling per mask).
        n_bins: Histogram bins for AUROC/AP.
        aupro_steps, expect_fpr: CFLOW-AD AUPRO parameters.
    """

    def __init__(self, lo, hi, aupro=True, n_bins=2 ** 20, aupro_steps=1000, expect_fpr=0.3):
        self.lo, self.hi = lo, hi
        self.n_bins = n_bins
        self.pos = np.zeros(n_bins, dtype=np.int64)
        self.neg = np.zeros(n_bins, dtype=np.int64)
        self.aupro = aupro
        self.expect_fpr = expect_fpr
        self.steps = aupro_steps
        delta = (hi - lo) / aupro_steps
        self.thresholds = (hi - np.arange(aupro_steps) * delta)[::-1]
        self.false_pos = np.zeros(aupro_steps)
        self.region_overlap = np.zeros(aupro_steps)
        self.n_regions = 0
        self.n_negative = 0

    def _bins(self, values):
        span = self.hi - self.lo
        if span <= 0:
            return np.zeros(values.size, dtype=np.int64)
        idx = ((values - self.lo) / span * self.n_bins).astype(np.int64)
        return np.clip(idx, 0, self.n_bins - 1)

    def _positives_per_step(self, values):
        below = np.searchsorted(self.thresholds, values, side='left')
        suffix = np.bincount(below, minlength=self.steps + 1)[::-1].cumsum()[::-1]
        return suffix[1:][::-1].astype(np.float64)

    def add(self, amap, mask):
        """Accumulate one image: ``amap`` and ``mask`` shaped ``(H, W)``."""
        amap = np.asarray(amap, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        values = amap.reshape(-1)
        labels = mask.reshape(-1)
        bins = self._bins(values)
        self.pos += np.bincount(bins[labels], minlength=self.n_bins)
        self.neg += np.bincount(bins[~labels], minlength=self.n_bins)

        if not self.aupro:
            return
        negative = amap[~mask]
        self.n_negative += negative.size
        self.false_pos += self._positives_per_step(negative)
        for rows, cols, area in aupro_regions(mask):
            self.region_overlap += self._positives_per_step(amap[rows, cols]) / area
            self.n_regions += 1

    def result(self):
        """``dict`` with ``pixel_auroc``, ``pixel_ap`` and ``aupro`` (NaN when undefined)."""
        n_pos, n_neg = self.pos.sum(), self.neg.sum()
        out = dict(pixel_auroc=float('nan'), pixel_ap=float('nan'), aupro=float('nan'))
        if n_pos == 0 or n_neg == 0:
            return out
        # AUROC = P(positive score > negative score) + 0.5 * P(tie in the same bin).
        neg_below = np.cumsum(self.neg) - self.neg
        out['pixel_auroc'] = float((self.pos * neg_below).sum() + 0.5 * (self.pos * self.neg).sum()) \
            / (float(n_pos) * float(n_neg))
        # AP: thresholds in decreasing order, one per bin.
        tp = np.cumsum(self.pos[::-1])
        fp = np.cumsum(self.neg[::-1])
        nonzero = (tp + fp) > 0
        precision = np.where(nonzero, tp / np.maximum(tp + fp, 1), 0.0)
        recall = tp / n_pos
        out['pixel_ap'] = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
        if self.aupro:
            out['aupro'] = self._aupro()
        return out

    def _aupro(self):
        if self.n_regions == 0 or self.n_negative == 0 or self.hi <= self.lo:
            return float('nan')
        fprs = self.false_pos / self.n_negative
        pros = self.region_overlap / self.n_regions
        keep = fprs <= self.expect_fpr
        fprs, pros = fprs[keep], pros[keep]
        if fprs.size < 2 or fprs.max() <= fprs.min():
            return float('nan')
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
        return float(auc(fprs, pros))


def aupro_regions(mask):
    """Ground-truth regions as the official CFLOW-AD AUPRO sees them.

    8-connected components; the overlap is counted on the hole-filled crop and
    divided by the unfilled area.

    Returns:
        List of ``(rows, cols, area)``.
    """
    if not mask.any():
        return []
    regions = []
    for prop in regionprops(label_regions(mask, connectivity=2)):
        r0, c0, _, _ = prop.bbox
        rows, cols = np.nonzero(binary_fill_holes(prop.image))
        regions.append((rows + r0, cols + c0, int(prop.area)))
    return regions


def image_metrics(labels, scores, prefix):
    """Image AUROC and AP; ``{}`` when only one class is present."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return {}
    return {f'{prefix}auroc': float(roc_auc_score(labels, scores)),
            f'{prefix}ap': float(average_precision_score(labels, scores))}
