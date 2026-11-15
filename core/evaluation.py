"""Test-set evaluation shared by training, the level ablation and ``evaluate.py``.

The pixel head runs once per image and the per-level NLL is cached
(:func:`cache_nll`, ~110 KB per image). Maps for any recipe or level subset are
recomputed from that cache without another forward pass.
"""

import numpy as np
import torch
from tqdm import tqdm

import config as c
from core import data
from core.metrics import PixelMetrics, ScoreRange, image_metrics
from core.model import image_anomaly_score
from core.pixel_head import Cohort, image_score_from_map, score_map


@torch.no_grad()
def image_head_scores(model, testset, device, desc='image score'):
    """SEDifferNet score per image; ``testset`` must provide ``n_views`` fixed rotations."""
    scores = []
    for views, _ in tqdm(data.loader(testset), desc=desc, leave=False):
        scores.append(image_anomaly_score(model, data.all_views(views, device)))
    return np.array(scores)


@torch.no_grad()
def cache_nll(head, dataset, tta=None, desc='NLL'):
    """Per-level NLL of every image: ``(N, L, out, out)`` in float32."""
    cache = []
    for views, _ in tqdm(data.loader(dataset), desc=desc, leave=False):
        cache.append(head.nll(data.first_view(views, head.device), tta=tta)[0].numpy())
    return np.stack(cache)


def load_masks(testset):
    """Masks of the anomalous images (``None`` for normal ones, whose mask is empty)."""
    return [testset.load_mask(i) if label else None for i, label in enumerate(testset.labels)]


def iter_maps(nll_cache, recipe, trained_levels, level_stats=None, cohort=None):
    """Yield the map of every image in the cohort for the given recipe."""
    cohort = cohort or Cohort.from_cache(nll_cache, recipe, trained_levels)
    for nll in nll_cache:
        yield score_map(nll, recipe, trained_levels, level_stats, cohort)


def evaluate_maps(nll_cache, masks, labels, recipe, trained_levels, level_stats=None,
                  aupro=True, desc=None):
    """Pixel metrics and the image score derived from the map.

    Args:
        nll_cache: Output of :func:`cache_nll`.
        masks: Output of :func:`load_masks`.
        labels: 0/1 label per image.
        recipe: Inference recipe.
        trained_levels: Levels of the head.
        level_stats: Statistics for ``per_level_std``.
        aupro: Also compute AUPRO.

    Returns:
        ``(metrics, map_scores)``.
    """
    cohort = Cohort.from_cache(nll_cache, recipe, trained_levels)
    score_range = ScoreRange()
    for amap in iter_maps(nll_cache, recipe, trained_levels, level_stats, cohort):
        score_range.add(amap)

    metrics = PixelMetrics(score_range.lo, score_range.hi, aupro=aupro)
    empty = np.zeros(c.img_size, dtype=bool)
    map_scores = []
    maps = iter_maps(nll_cache, recipe, trained_levels, level_stats, cohort)
    for amap, mask in tqdm(zip(maps, masks), total=len(masks), desc=desc, leave=False,
                           disable=desc is None):
        metrics.add(amap, empty if mask is None else mask)
        map_scores.append(image_score_from_map(amap, recipe['image_score'],
                                               recipe.get('topk_percent', 1.0)))
    result = metrics.result()
    result.update(image_metrics(labels, map_scores, prefix='image_from_map_'))
    return result, np.array(map_scores)
