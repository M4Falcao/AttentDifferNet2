"""Apply both heads of a class to a folder of images.

For each image it saves the image-head score (SEDifferNet, 64 fixed rotations),
the score derived from the pixel map, the anomaly map (``.npy``) and an overlay
(``.png``). The inference recipe is the one stored in the pixel-head checkpoint;
``--recipe`` replaces it.

Examples::

    python infer.py --classes vari-grip --input <image folder>
    python infer.py --classes vari-grip --input <folder> --model_dir runs/20261112_120000
    python infer.py --classes vari-grip --input <folder> --image_head <se.pt> --pixel_head <head.pt>

Note: with ``score_mode = cflow_original`` the map is normalized by the cohort,
i.e. by the images of the input folder.
"""

import argparse
import csv
import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config as c
from core.inference import AnomalyDetector, denormalize, folder_datasets
from core.recipes import resolve_recipe
from core.utils import resolve_device, resolve_models, setup_console


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--classes', nargs='+', required=True,
                        help='class(es) whose models are used')
    parser.add_argument('--input', required=True, help='image folder')
    parser.add_argument('--output', default=c.infer_output_dir)
    parser.add_argument('--model_dir', default=c.model_dir,
                        help='training run (default: the most recent in config.output_dir)')
    parser.add_argument('--image_head', default=None, help='explicit SEDifferNet checkpoint')
    parser.add_argument('--pixel_head', default=None, help='explicit pixel-head checkpoint')
    parser.add_argument('--recipe', default=None, choices=sorted(c.PIXEL_RECIPES),
                        help='replaces the inference recipe stored in the checkpoint')
    parser.add_argument('--dataset', default=c.dataset_path,
                        help='only used to resolve --recipe (recipes are planned per dataset)')
    parser.add_argument('--device', default=c.device)
    args = parser.parse_args()
    if (args.image_head or args.pixel_head) and len(args.classes) != 1:
        parser.error('--image_head/--pixel_head require exactly one class')
    return args


def save_overlay(path, image, amap):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title('image')
    axes[1].imshow(image)
    axes[1].imshow(amap, cmap='jet', alpha=0.5)
    axes[1].set_title('anomaly map')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    setup_console()
    args = parse_args()
    device = resolve_device(args.device)
    rotations, single_view = folder_datasets(args.input)

    for class_name in args.classes:
        image_head, pixel_head = resolve_models(class_name, args.model_dir, c.output_dir,
                                                args.image_head, args.pixel_head)
        override = None
        if args.recipe:
            override = resolve_recipe(args.dataset, class_name, args.recipe)['inference']
            print(f"WARNING: checkpoint recipe replaced by {args.recipe}; it is only the measured "
                  f"one if the head was trained with the same variant.")
        detector = AnomalyDetector(image_head, pixel_head, device, override)
        print(f"{class_name}: {detector.describe()}")

        out_dir = os.path.join(args.output, class_name)
        os.makedirs(os.path.join(out_dir, 'heatmaps'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'overlays'), exist_ok=True)

        image_scores = detector.image_scores(rotations)
        nll = detector.nll(single_view)
        rows = []
        for i, amap in enumerate(detector.maps(nll)):
            name = os.path.splitext(os.path.basename(single_view.paths[i]))[0]
            np.save(os.path.join(out_dir, 'heatmaps', f'{name}.npy'), amap.astype(np.float32))
            save_overlay(os.path.join(out_dir, 'overlays', f'{name}.png'),
                         denormalize(single_view[i][0][0]), amap)
            rows.append(dict(image=single_view.paths[i], image_score=image_scores[i],
                             map_score=detector.map_score(amap)))

        with open(os.path.join(out_dir, 'scores.csv'), 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=['image', 'image_score', 'map_score'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {len(rows)} images -> {out_dir}")


if __name__ == '__main__':
    main()
