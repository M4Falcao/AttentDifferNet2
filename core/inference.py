"""Complete detector: image head + pixel head of one class."""

import numpy as np

import config as c
from core import data
from core.evaluation import cache_nll, image_head_scores, iter_maps
from core.model import load_image_head
from core.pixel_head import image_score_from_map, load_pixel_head


class AnomalyDetector:
    """Load both heads of a class and apply the recipe stored in the checkpoint.

    Args:
        image_head_path: ``image_head.pt`` (or an older SEDifferNet checkpoint).
        pixel_head_path: ``pixel_head.pt`` (or an older ``best_pixel_auroc.pt``).
        device: Torch device.
        recipe_override: Inference recipe that replaces the one in the checkpoint.
    """

    def __init__(self, image_head_path, pixel_head_path, device, recipe_override=None):
        self.device = device
        self.image_model = load_image_head(image_head_path, device)
        self.pixel_head, self.checkpoint = load_pixel_head(pixel_head_path, self.image_model,
                                                           device, recipe_override)
        self.recipe = self.pixel_head.recipe

    def describe(self):
        trained = self.checkpoint.get('training_recipe') or {}
        return (f"training recipe {trained.get('name', '?')} | inference: "
                f"score {self.recipe['score_mode']}, levels {self.recipe['levels']}, "
                f"pad {self.recipe['pad']} ({self.recipe['pad_mode']}), tta {self.recipe['tta']}, "
                f"sigma {self.recipe['sigma']}, image score {self.recipe['image_score']}")

    def image_scores(self, dataset_with_rotations):
        """SEDifferNet score per image (dataset with ``n_views`` fixed rotations)."""
        return image_head_scores(self.image_model, dataset_with_rotations, self.device)

    def nll(self, dataset):
        """Per-level NLL of every image (dataset with one unaugmented view)."""
        return cache_nll(self.pixel_head, dataset)

    def maps(self, nll_cache):
        """Yield the anomaly maps of the cohort (``cflow_original`` uses the whole cohort)."""
        return iter_maps(nll_cache, self.recipe, self.pixel_head.trained_levels,
                         self.pixel_head.level_stats)

    def map_score(self, amap):
        return image_score_from_map(amap, self.recipe['image_score'],
                                    self.recipe.get('topk_percent', 1.0))


def folder_datasets(folder):
    """Datasets of an image folder: fixed rotations (image) and a single view (pixel)."""
    return data.FolderDataset(folder, n_views=c.n_transforms_test), data.FolderDataset(folder)


def denormalize(tensor):
    """Normalized tensor ``(3, H, W)`` -> image ``(H, W, 3)`` in [0, 1]."""
    image = tensor.permute(1, 2, 0).cpu().numpy() * np.array(c.norm_std) + np.array(c.norm_mean)
    return np.clip(image, 0, 1)
