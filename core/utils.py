"""Small utilities: device, seed, JSON and locating the models of a run."""

import json
import os
import random
import sys

import numpy as np
import torch

IMAGE_HEAD_FILE = 'image_head.pt'
PIXEL_HEAD_FILE = 'pixel_head.pt'


def setup_console():
    """UTF-8 output (the default Windows console uses cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')


def resolve_device(name):
    """``'cuda'`` falls back to ``'cpu'`` when no GPU is available."""
    if name.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        return 'cpu'
    return name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and value != value:
        return None
    return value


def save_json(path, payload):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(_to_jsonable(payload), fh, indent=2, ensure_ascii=False)


def class_checkpoints(model_dir, class_name):
    """Paths ``(image_head.pt, pixel_head.pt)`` of a class in a training run."""
    class_dir = os.path.join(model_dir, class_name)
    image_head = os.path.join(class_dir, IMAGE_HEAD_FILE)
    pixel_head = os.path.join(class_dir, PIXEL_HEAD_FILE)
    missing = [p for p in (image_head, pixel_head) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"models of {class_name} not found: {missing}")
    return image_head, pixel_head


def latest_run(output_dir):
    """Most recent run in ``output_dir`` (timestamped folders)."""
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"runs folder not found: {output_dir}")
    runs = sorted(d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)))
    if not runs:
        raise FileNotFoundError(f"no runs in {output_dir}")
    return os.path.join(output_dir, runs[-1])


def resolve_models(class_name, model_dir=None, output_dir=None, image_head=None, pixel_head=None):
    """Paths of both heads of a class.

    Priority: explicit paths (``image_head``/``pixel_head``) > ``model_dir`` >
    most recent run in ``output_dir``.
    """
    if image_head or pixel_head:
        if not (image_head and pixel_head):
            raise ValueError('pass --image_head and --pixel_head together')
        return image_head, pixel_head
    return class_checkpoints(model_dir or latest_run(output_dir), class_name)
