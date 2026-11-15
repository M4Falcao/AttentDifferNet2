"""Quick level ablation of the pixel head.

Runs at the end of training when the dataset or the class has no planned recipe.
With the per-level NLL already cached, the maps of every non-empty subset of the
trained levels are recomputed without another forward pass, and the selected
subset goes into the inference recipe stored in the checkpoint.

Criterion: the subset with the highest Pixel AUROC only replaces "all levels" if
it gains at least ``min_gain``. This avoids changing the recipe for differences
of the size of sampling noise (all levels was the minimax choice across the 5
INSPLAD classes).
"""

from itertools import combinations

from core.evaluation import evaluate_maps
from core.pixel_head import levels_to_str


def level_subsets(trained_levels):
    """All non-empty subsets of the trained levels, as strings (``"0"``, ``"02"``, ...)."""
    subsets = []
    for size in range(1, len(trained_levels) + 1):
        subsets += [levels_to_str(combo) for combo in combinations(trained_levels, size)]
    return subsets


def run_level_ablation(nll_cache, masks, labels, recipe, trained_levels, level_stats, min_gain):
    """Compare the level subsets and pick the one that goes into the recipe.

    Returns:
        ``dict`` with ``status``, ``selected`` (level string), ``reference``,
        ``min_gain``, ``selected_on``, ``n_images``, ``n_anomalous`` and ``table``
        (one row per subset with its Pixel AUROC).
    """
    reference = levels_to_str(trained_levels)
    summary = dict(reference=reference, selected=reference, min_gain=min_gain,
                   selected_on='test', n_images=int(len(labels)),
                   n_anomalous=int(sum(labels)), table=[])
    if not any(mask is not None and mask.any() for mask in masks):
        summary['status'] = 'skipped: no ground-truth masks in the test set'
        return summary

    for subset in level_subsets(trained_levels):
        metrics, _ = evaluate_maps(nll_cache, masks, labels, dict(recipe, levels=subset),
                                   trained_levels, level_stats, aupro=False)
        summary['table'].append(dict(levels=subset, pixel_auroc=metrics['pixel_auroc']))
        print(f"    levels {subset:<3}  Pixel AUROC {metrics['pixel_auroc']:.4f}")

    by_level = {row['levels']: row['pixel_auroc'] for row in summary['table']}
    best = max(summary['table'], key=lambda row: row['pixel_auroc'])
    if best['pixel_auroc'] >= by_level[reference] + min_gain:
        summary['selected'] = best['levels']
    summary['status'] = 'done'
    return summary
