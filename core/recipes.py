"""Resolve the pixel-head recipe for a dataset and a class.

The named recipes in ``config.PIXEL_RECIPES`` only apply to the dataset they were
measured on (``config.planned_dataset``) and to the classes listed in them.
Everything else gets ``config.FALLBACK_RECIPE`` and training ends with the level
ablation (:mod:`core.level_ablation`).
"""

import os

import config as c
from core.pixel_head import levels_to_str


def dataset_identity(dataset_path):
    """Dataset folder name, without trailing separators, lower-cased."""
    return os.path.basename(os.path.normpath(dataset_path)).lower()


def resolve_recipe(dataset_path, class_name, name=None, level_ablation=None):
    """Complete recipe of a class.

    Args:
        dataset_path: Dataset root (decides whether a planned recipe exists).
        class_name: Class.
        name: Recipe name (default: ``config.pixel_recipe``).
        level_ablation: ``'auto'``, ``'always'`` or ``'never'`` (default: ``config.level_ablation``).

    Returns:
        ``dict`` with ``name``, ``planned`` (bool), ``class_name``, ``training``,
        ``inference``, ``epochs``, ``eval_interval`` and ``level_ablation`` (bool).
    """
    name = name or c.pixel_recipe
    if name not in c.PIXEL_RECIPES:
        raise ValueError(f"unknown recipe {name!r}; options: {sorted(c.PIXEL_RECIPES)}")

    preset = c.PIXEL_RECIPES[name]
    planned = (dataset_identity(dataset_path) == c.planned_dataset.lower()
               and class_name in c.PIXEL_EPOCHS
               and (not preset['per_class'] or class_name in preset['per_class']))

    training = dict(c.PIXEL_TRAINING_DEFAULTS)
    inference = dict(c.PIXEL_INFERENCE_DEFAULTS)
    if planned:
        per_class = preset['per_class'].get(class_name, {})
        training.update(preset['training'], **per_class.get('training', {}))
        inference.update(preset['inference'], **per_class.get('inference', {}))
        epochs, eval_interval = c.PIXEL_EPOCHS[class_name]
        resolved_name = name
    else:
        training.update(c.FALLBACK_RECIPE['training'])
        inference.update(c.FALLBACK_RECIPE['inference'])
        epochs, eval_interval = c.FALLBACK_EPOCHS
        resolved_name = 'fallback'

    level_ablation = level_ablation or c.level_ablation
    if level_ablation not in ('auto', 'always', 'never'):
        raise ValueError("level_ablation must be 'auto', 'always' or 'never'")
    ablation = level_ablation == 'always' or (level_ablation == 'auto' and not planned)

    training['levels'] = levels_to_str(training['levels'])
    inference['levels'] = levels_to_str(inference['levels'])
    return dict(name=resolved_name, requested=name, planned=planned, class_name=class_name,
                training=training, inference=inference, epochs=epochs,
                eval_interval=eval_interval, level_ablation=ablation)
