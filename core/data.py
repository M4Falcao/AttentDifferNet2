"""Datasets and transforms.

Expected layout (MVTec): ``<root>/<class>/train/good``,
``<root>/<class>/test/<good|defect>`` and ``<root>/<class>/ground_truth/<defect>``.

Every dataset returns the images as a stack of views ``(V, 3, H, W)``:

- training: ``V`` random views (rotation/jitter from ``config``);
- test, image level: ``V`` evenly spaced fixed rotations (64 by default), which
  make the SEDifferNet score deterministic;
- test, pixel level: one unaugmented view (``V = 1``), aligned with the mask.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.transforms.functional import rotate

import config as c

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ═══════════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════════

def _color_jitter():
    if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
        return [transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                       saturation=c.transf_saturation)]
    return []


def _to_normalized_tensor():
    return [transforms.ToTensor(), transforms.Normalize(c.norm_mean, c.norm_std)]


def train_transform(rotations=None):
    """Random training transform: resize, optional rotation, optional jitter."""
    rotations = c.transf_rotations if rotations is None else rotations
    augment = [transforms.RandomRotation(180)] if rotations else []
    return transforms.Compose([transforms.Resize(c.img_size)] + augment + _color_jitter()
                              + _to_normalized_tensor())


def rotation_transform(degrees):
    """Deterministic transform with one fixed rotation (SEDifferNet image score)."""
    rotation = transforms.Lambda(lambda img: rotate(img, degrees, False, False, None))
    return transforms.Compose([transforms.Resize(c.img_size), rotation] + _color_jitter()
                              + _to_normalized_tensor())


def eval_transform():
    """Resize + normalization, no augmentation. Required for pixel metrics."""
    return transforms.Compose([transforms.Resize(c.img_size)] + _to_normalized_tensor())


def mask_transform():
    return transforms.Compose([
        transforms.Resize(c.img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


def _stack_views(image, n_views, fixed_rotations, transform):
    """Apply the transform ``n_views`` times (or the fixed rotations) and stack the views."""
    if fixed_rotations:
        views = [rotation_transform(i * 360.0 / n_views)(image) for i in range(n_views)]
    else:
        views = [transform(image) for _ in range(n_views)]
    return torch.stack(views, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════════════

class TrainDataset(Dataset):
    """Normal images from ``<root>/<class>/train``.

    Args:
        dataset_path: Dataset root.
        class_name: Class.
        n_views: Views per image.
        transform: Transform applied to each view (default: :func:`train_transform`).
    """

    def __init__(self, dataset_path, class_name, n_views=1, transform=None):
        root = os.path.join(dataset_path, class_name, 'train')
        if not os.path.isdir(root):
            raise FileNotFoundError(f"training folder not found: {root}")
        self.paths = []
        for sub in sorted(os.listdir(root)):
            sub_dir = os.path.join(root, sub)
            if not os.path.isdir(sub_dir):
                continue
            for dirpath, _, files in sorted(os.walk(sub_dir, followlinks=True)):
                for fname in sorted(files):
                    if fname.lower().endswith(IMG_EXTENSIONS):
                        self.paths.append(os.path.join(dirpath, fname))
        if not self.paths:
            raise FileNotFoundError(f"no images in {root}")
        self.n_views = n_views
        self.transform = transform or train_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(self.paths[index])
        return _stack_views(image, self.n_views, False, self.transform), 0


class TestDataset(Dataset):
    """Images from ``<root>/<class>/test`` with label and mask.

    ``good`` folders get label 0 and an empty mask; every other folder gets label 1
    and the mask from ``ground_truth/<defect>/`` with the same file name (the
    ``_mask``/``_gt`` suffixes and other extensions are also accepted).

    Args:
        dataset_path: Dataset root.
        class_name: Class.
        n_views: ``1`` = one unaugmented view (pixel level); ``>1`` = evenly spaced
            fixed rotations (image score).
    """

    def __init__(self, dataset_path, class_name, n_views=1):
        test_dir = os.path.join(dataset_path, class_name, 'test')
        gt_dir = os.path.join(dataset_path, class_name, 'ground_truth')
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"test folder not found: {test_dir}")
        if 'good' not in os.listdir(test_dir):
            raise FileNotFoundError(f"{test_dir} must contain a 'good' subfolder")

        self.samples = []   # (path, label 0/1, mask path or None)
        for defect in sorted(os.listdir(test_dir)):
            defect_dir = os.path.join(test_dir, defect)
            if not os.path.isdir(defect_dir):
                continue
            label = 0 if defect == 'good' else 1
            for fname in sorted(os.listdir(defect_dir)):
                path = os.path.join(defect_dir, fname)
                if not (os.path.isfile(path) and fname.lower().endswith(IMAGE_EXTENSIONS)):
                    continue
                mask = _find_mask(os.path.join(gt_dir, defect), fname) if label else None
                self.samples.append((path, label, mask))

        self.labels = np.array([label for _, label, _ in self.samples], dtype=np.uint8)
        self.n_views = n_views
        self._eval = eval_transform()
        self._mask = mask_transform()

    def __len__(self):
        return len(self.samples)

    @property
    def has_masks(self):
        return any(mask is not None for _, _, mask in self.samples)

    def load_mask(self, index):
        """Binary ``(H, W)`` mask as ``bool``."""
        _, _, mask_path = self.samples[index]
        if mask_path is None:
            return np.zeros(c.img_size, dtype=bool)
        return self._mask(Image.open(mask_path).convert('L'))[0].numpy() > 0.5

    def __getitem__(self, index):
        path, label, _ = self.samples[index]
        image = Image.open(path).convert('RGB')
        views = _stack_views(image, self.n_views, self.n_views > 1, self._eval)
        return views, label


class FolderDataset(Dataset):
    """Loose images from a folder (inference), as one unaugmented view or fixed rotations."""

    def __init__(self, folder, n_views=1):
        self.paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                            if f.lower().endswith(IMAGE_EXTENSIONS))
        if not self.paths:
            raise FileNotFoundError(f"no images in {folder}")
        self.n_views = n_views
        self._eval = eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        return _stack_views(image, self.n_views, self.n_views > 1, self._eval), 0


def _find_mask(gt_class_dir, fname):
    if not os.path.isdir(gt_class_dir):
        return None
    candidate = os.path.join(gt_class_dir, fname)
    if os.path.exists(candidate):
        return candidate
    stem = os.path.splitext(fname)[0]
    for mask_name in os.listdir(gt_class_dir):
        if os.path.splitext(mask_name)[0] in (stem, stem + '_mask', stem + '_gt'):
            return os.path.join(gt_class_dir, mask_name)
    return None


def loader(dataset, shuffle=False):
    """DataLoader with the ``config`` batch size (training) or 1 (evaluation)."""
    return DataLoader(dataset, batch_size=c.batch_size if shuffle else 1, shuffle=shuffle,
                      num_workers=c.num_workers, pin_memory=torch.cuda.is_available())


def first_view(images, device):
    """``(B, V, 3, H, W)`` -> ``(B, 3, H, W)`` with the first view, on the device."""
    return images[:, 0].to(device)


def all_views(images, device):
    """``(B, V, 3, H, W)`` -> ``(B * V, 3, H, W)``, on the device."""
    return images.to(device).view(-1, *images.shape[-3:])
