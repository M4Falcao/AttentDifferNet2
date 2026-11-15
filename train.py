"""Train both heads of each class: image head (SEDifferNet), then pixel head.

All defaults come from ``config.py``; the arguments below override them for this
run only.

Examples::

    python train.py                                   # every class in config.classes
    python train.py --classes vari-grip               # one class
    python train.py --classes vari-grip --recipe general
    python train.py --dataset <folder> --classes <class>   # dataset without a recipe: fallback + level ablation
    python train.py --classes vari-grip --image_head <checkpoint>   # skip the image head
    python train.py --classes vari-grip --print_recipe    # only print the recipe

Output: ``<output_dir>/<timestamp>/<class>/`` with ``image_head.pt``, ``pixel_head.pt``,
``metrics.json`` and training histories; ``summary.csv`` and ``config_snapshot.json``
at the run level.
"""

import argparse
import csv
import json
import os
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import config as c
from core.data import TestDataset
from core.evaluation import image_head_scores
from core.metrics import image_metrics
from core.model import load_image_head
from core.recipes import resolve_recipe
from core.tracking import Tracker
from core.train_image import train_image_head
from core.train_pixel import train_pixel_head
from core.utils import resolve_device, save_json, set_seed, setup_console


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--classes', nargs='+', default=None,
                        help='classes to train (default: config.classes)')
    parser.add_argument('--dataset', default=c.dataset_path, help='dataset root')
    parser.add_argument('--recipe', default=c.pixel_recipe, choices=sorted(c.PIXEL_RECIPES),
                        help='pixel-head recipe')
    parser.add_argument('--image_head', default=None,
                        help='already trained SEDifferNet (single class only); skips image training')
    parser.add_argument('--output_dir', default=c.output_dir)
    parser.add_argument('--level_ablation', default=c.level_ablation,
                        choices=['auto', 'always', 'never'])
    parser.add_argument('--device', default=c.device)
    parser.add_argument('--mlflow', action='store_true', default=c.use_mlflow,
                        help='log the run to MLflow')
    parser.add_argument('--image_epochs', nargs=2, type=int, metavar=('PRE', 'META'), default=None,
                        help='image-head epochs (default: config.pre_epochs/meta_epochs)')
    parser.add_argument('--pixel_epochs', type=int, default=None,
                        help='pixel-head epochs (default: those of the recipe)')
    parser.add_argument('--print_recipe', action='store_true',
                        help='print the resolved recipe of each class and exit')
    args = parser.parse_args()
    args.classes = args.classes or list(c.classes)
    if args.image_head and len(args.classes) != 1:
        parser.error('--image_head requires exactly one class in --classes')
    return args


def config_snapshot(args):
    values = {k: v for k, v in vars(c).items()
              if not k.startswith('_') and not callable(v) and not isinstance(v, type(os))}
    return dict(config=values, arguments=vars(args))


def train_class(class_name, args, run_dir, device):
    set_seed(c.seed)
    out_dir = os.path.join(run_dir, class_name)
    os.makedirs(out_dir, exist_ok=True)
    recipe = resolve_recipe(args.dataset, class_name, args.recipe, args.level_ablation)
    tracker = Tracker(args.mlflow, run_name=f"{class_name}_{os.path.basename(run_dir)}",
                      params=dict(class_name=class_name, dataset=args.dataset, recipe=recipe['name']))
    try:
        image_head = args.image_head or c.image_head_checkpoints.get(class_name)
        if image_head:
            print(f"  image head: using {image_head}")
            model = load_image_head(image_head, device)
            scores = image_head_scores(model, TestDataset(args.dataset, class_name,
                                                          n_views=c.n_transforms_test), device)
            image_auroc = image_metrics(TestDataset(args.dataset, class_name).labels, scores,
                                        prefix='').get('auroc', float('nan'))
        else:
            image_head, image_auroc = train_image_head(class_name, args.dataset, out_dir, device,
                                                       tracker, epochs=args.image_epochs)
            model = load_image_head(image_head, device)

        _, metrics, ablation = train_pixel_head(model, image_head, recipe, args.dataset, out_dir,
                                                device, tracker, epochs=args.pixel_epochs)
        metrics = dict(class_name=class_name, recipe=recipe['name'],
                       inference_levels=ablation['selected'] if ablation else recipe['inference']['levels'],
                       image_auroc=image_auroc, **metrics)
        save_json(os.path.join(out_dir, 'metrics.json'), metrics)
        for key in ('image_auroc', 'pixel_auroc', 'pixel_ap', 'aupro'):
            tracker.metric(f'final_{key}', metrics[key])
        return metrics
    finally:
        tracker.close()


def main():
    setup_console()
    args = parse_args()
    if args.print_recipe:
        for class_name in args.classes:
            recipe = resolve_recipe(args.dataset, class_name, args.recipe, args.level_ablation)
            print(json.dumps(recipe, indent=2, ensure_ascii=False))
        return

    device = resolve_device(args.device)
    run_dir = os.path.join(args.output_dir, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    save_json(os.path.join(run_dir, 'config_snapshot.json'), config_snapshot(args))
    print(f"Run: {run_dir}\nDataset: {args.dataset}\nDevice: {device}")

    rows = []
    for i, class_name in enumerate(args.classes, 1):
        print(f"\n[{i}/{len(args.classes)}] {class_name}")
        metrics = train_class(class_name, args, run_dir, device)
        rows.append(metrics)
        print(f"  Image AUROC {metrics['image_auroc']:.4f} | Pixel AUROC {metrics['pixel_auroc']:.4f}"
              f" | Pixel AP {metrics['pixel_ap']:.4f} | AUPRO {metrics['aupro']:.4f}")

    columns = ['class_name', 'recipe', 'inference_levels', 'image_auroc', 'image_from_map_auroc',
               'pixel_auroc', 'pixel_ap', 'aupro', 'selection_epoch']
    with open(os.path.join(run_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary: {os.path.join(run_dir, 'summary.csv')}")


if __name__ == '__main__':
    main()
