"""Equivalence with the code before the refactoring.

Compares image scores, NLL and pixel maps with reference outputs produced by the
previous code on 16 images per class (8 anomalous + 8 normal) and 3 inference
recipes. Needs the local checkpoints and reference outputs; without them the test
is skipped.

    GABARITO_DIR=<folder with manifest.json and <class>.npz> python -m pytest tests/test_equivalence.py
"""

import json
import os

import numpy as np
import pytest
import torch

import config as c
from core import data
from core.model import image_anomaly_score, load_image_head
from core.pixel_head import Cohort, load_pixel_head, score_map

GABARITO = os.environ.get('GABARITO_DIR')
DATASET = os.environ.get('GABARITO_DATASET', c.dataset_path)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

pytestmark = pytest.mark.skipif(not GABARITO or not os.path.isdir(GABARITO or ''),
                                reason='GABARITO_DIR is not set')


def _manifest():
    with open(os.path.join(GABARITO, 'manifest.json'), encoding='utf-8') as fh:
        return json.load(fh)


@pytest.mark.parametrize('class_name', ['glass-insulator', 'lightning-rod-suspension',
                                        'polymer-insulator-upper-shackle', 'vari-grip',
                                        'yoke-suspension'])
def test_class_matches_reference(class_name):
    entry = _manifest()[class_name]
    ref = np.load(os.path.join(GABARITO, f'{class_name}.npz'))

    pixel_set = data.TestDataset(DATASET, class_name, n_views=1)
    image_set = data.TestDataset(DATASET, class_name, n_views=64)
    rel = [os.path.relpath(p, DATASET) for p, _, _ in pixel_set.samples]
    idx = [rel.index(p) for p in entry['paths']]

    model = load_image_head(entry['se'], DEVICE)
    scores = np.array([image_anomaly_score(model, image_set[i][0].to(DEVICE)) for i in idx])
    np.testing.assert_allclose(scores, ref['image_scores'], rtol=1e-4)

    masks = np.stack([pixel_set.load_mask(i) for i in idx])
    np.testing.assert_array_equal(masks, ref['masks'] > 0.5)

    images = torch.stack([pixel_set[i][0][0] for i in idx])
    for name, recipe in entry['recipes'].items():
        head, _ = load_pixel_head(entry['head'], model, DEVICE, recipe_override=recipe)
        nll = np.stack([head.nll(images[b:b + 1].to(DEVICE))[0].numpy() for b in range(len(idx))])
        np.testing.assert_allclose(nll, ref[f'{name}_nll'], rtol=1e-4, atol=1e-3)

        stats = [tuple(s) for s in entry['level_stats']]
        cohort = Cohort.from_cache(nll, head.recipe, head.trained_levels)
        maps = np.stack([score_map(n, head.recipe, head.trained_levels, stats, cohort) for n in nll])
        expected = ref[f'{name}_maps']
        scale = np.abs(expected).max()
        np.testing.assert_allclose(maps, expected, rtol=1e-4, atol=1e-4 * scale,
                                   err_msg=f'{class_name}/{name}')
