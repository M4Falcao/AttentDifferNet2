"""Evaluate both heads on the test set of each class.

Metrics: SEDifferNet Image AUROC/AP (64 fixed rotations), Image AUROC/AP of the
map-derived score, Pixel AUROC, Pixel AP and AUPRO (CFLOW-AD definition). The
inference recipe is the one stored in the pixel-head checkpoint; ``--recipe``
replaces it.

Examples::

    python evaluate.py                                   # most recent run, every class
    python evaluate.py --model_dir runs/20261112_120000 --classes vari-grip
    python evaluate.py --classes vari-grip --image_head <se.pt> --pixel_head <head.pt>

Output: ``<output>/<timestamp>/`` with ``summary.csv``, ``<class>_metrics.json`` and
``<class>_scores.csv`` (per-image scores).
"""

import argparse
import csv
import os
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import config as c
from core.data import TestDataset
from core.evaluation import evaluate_maps, load_masks
from core.inference import AnomalyDetector
from core.metrics import image_metrics
from core.recipes import resolve_recipe
from core.utils import resolve_device, resolve_models, save_json, setup_console

COLUMNS = ['class_name', 'n_images', 'n_anomalous', 'image_auroc', 'image_ap',
           'image_from_map_auroc', 'image_from_map_ap', 'pixel_auroc', 'pixel_ap', 'aupro',
           'score_mode', 'levels']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--classes', nargs='+', default=None,
                        help='classes to evaluate (default: config.classes)')
    parser.add_argument('--dataset', default=c.dataset_path)
    parser.add_argument('--model_dir', default=c.model_dir,
                        help='training run (default: the most recent in config.output_dir)')
    parser.add_argument('--image_head', default=None, help='explicit SEDifferNet checkpoint')
    parser.add_argument('--pixel_head', default=None, help='explicit pixel-head checkpoint')
    parser.add_argument('--recipe', default=None, choices=sorted(c.PIXEL_RECIPES),
                        help='replaces the inference recipe stored in the checkpoint')
    parser.add_argument('--output', default=c.eval_output_dir)
    parser.add_argument('--no_aupro', action='store_true', help='skip AUPRO (faster)')
    parser.add_argument('--device', default=c.device)
    args = parser.parse_args()
    args.classes = args.classes or list(c.classes)
    if (args.image_head or args.pixel_head) and len(args.classes) != 1:
        parser.error('--image_head/--pixel_head require exactly one class')
    return args


def evaluate_class(class_name, args, device, out_dir):
    image_head, pixel_head = resolve_models(class_name, args.model_dir, c.output_dir,
                                            args.image_head, args.pixel_head)
    override = None
    if args.recipe:
        override = resolve_recipe(args.dataset, class_name, args.recipe)['inference']
    detector = AnomalyDetector(image_head, pixel_head, device, override)
    print(f"  {detector.describe()}")

    pixel_set = TestDataset(args.dataset, class_name, n_views=1)
    rotation_set = TestDataset(args.dataset, class_name, n_views=c.n_transforms_test)
    labels = pixel_set.labels

    image_scores = detector.image_scores(rotation_set)
    nll = detector.nll(pixel_set)
    metrics, map_scores = evaluate_maps(nll, load_masks(pixel_set), labels, detector.recipe,
                                        detector.pixel_head.trained_levels,
                                        detector.pixel_head.level_stats,
                                        aupro=not args.no_aupro, desc='pixel metrics')
    metrics.update(image_metrics(labels, image_scores, prefix='image_'))
    metrics.update(class_name=class_name, n_images=int(len(labels)), n_anomalous=int(labels.sum()),
                   score_mode=detector.recipe['score_mode'], levels=detector.recipe['levels'],
                   image_head=image_head, pixel_head=pixel_head, recipe=detector.recipe)
    save_json(os.path.join(out_dir, f'{class_name}_metrics.json'), metrics)

    with open(os.path.join(out_dir, f'{class_name}_scores.csv'), 'w', newline='',
              encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['image', 'label', 'image_score', 'map_score'])
        for (path, label, _), s_img, s_map in zip(pixel_set.samples, image_scores, map_scores):
            writer.writerow([os.path.relpath(path, args.dataset), int(label), s_img, s_map])
    return metrics


def main():
    setup_console()
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = os.path.join(args.output, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for class_name in args.classes:
        print(f"\n{class_name}")
        metrics = evaluate_class(class_name, args, device, out_dir)
        rows.append(metrics)
        print(f"  Image AUROC {metrics.get('image_auroc', float('nan')):.4f} | "
              f"Pixel AUROC {metrics['pixel_auroc']:.4f} | Pixel AP {metrics['pixel_ap']:.4f} | "
              f"AUPRO {metrics['aupro']:.4f}")

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary: {os.path.join(out_dir, 'summary.csv')}")


if __name__ == '__main__':
    main()
