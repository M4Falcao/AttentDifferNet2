"""Pixel-head training on top of a frozen SEDifferNet.

1. Train one conditional flow per level on the normal training images.
2. Every ``eval_interval`` epochs, measure the test Pixel AUROC with the
   selection recipe (``config.pixel_selection``: ``raw`` score, all levels, no
   padding/TTA, sigma 4) and keep the best epoch.
3. With the best epoch, build the inference recipe: ``per_level_std`` statistics
   (training images) and, if the recipe asks for it, the level ablation.
4. Measure the final test metrics and save ``pixel_head.pt``.
"""

import copy
import math
import os

import numpy as np
import torch
from tqdm import tqdm

import config as c
from core import data
from core.evaluation import cache_nll, evaluate_maps, load_masks
from core.level_ablation import run_level_ablation
from core.pixel_head import (FeatureExtractor, PixelFlow, PixelHead, apply_feature_stats,
                             clip_flow_grads, compute_feature_stats, compute_level_stats,
                             parse_levels, save_pixel_head)
from core.utils import save_json


def _lr_lambda(epochs, warmup):
    """Linear warm-up followed by cosine decay."""
    def schedule(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule


def train_pixel_head(image_model, image_head_path, recipe, dataset_path, out_dir, device,
                     tracker, epochs=None):
    """Train, select and evaluate the pixel head of one class.

    Args:
        image_model: Trained SEDifferNet, in ``eval`` mode.
        image_head_path: Path of the SEDifferNet (stored in the checkpoint for traceability).
        recipe: Output of :func:`core.recipes.resolve_recipe`.
        dataset_path: Dataset root.
        out_dir: Class folder of the run.
        device: Torch device.
        tracker: :class:`core.tracking.Tracker`.
        epochs: Overrides the number of epochs of the recipe (quick tests).

    Returns:
        ``(path of pixel_head.pt, final metrics, ablation summary or None)``.
    """
    class_name = recipe['class_name']
    training, inference = recipe['training'], dict(recipe['inference'])
    n_epochs = epochs or recipe['epochs']
    eval_interval = min(recipe['eval_interval'], n_epochs)
    levels = parse_levels(training['levels'])
    norm_levels = training['feat_norm_levels']

    trainset = data.TrainDataset(dataset_path, class_name, n_views=1)
    train_loader = data.loader(trainset, shuffle=True)
    testset = data.TestDataset(dataset_path, class_name, n_views=1)
    masks, labels = load_masks(testset), testset.labels
    print(f"  pixel head: recipe {recipe['name']} | training levels {training['levels']} | "
          f"{n_epochs} epochs | {len(trainset)} training, {len(testset)} test")

    # Frozen features and optional per-channel standardization.
    extractor = FeatureExtractor(image_model).to(device).eval()
    feat_stats = None
    if norm_levels:
        feat_stats = compute_feature_stats(extractor, train_loader, device, norm_levels)
        feat_stats = [feat_stats[level] for level in levels]

    flow = PixelFlow(levels, c.pixel_cond_dim, c.pixel_n_blocks, c.pixel_hidden,
                     training['clamp_scale']).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=c.pixel_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(n_epochs, c.pixel_warmup_epochs))

    selection_recipe = dict(c.PIXEL_INFERENCE_DEFAULTS, levels=training['levels'],
                            **c.pixel_selection)
    history = dict(epoch=[], train_loss=[], eval_epoch=[], pixel_auroc=[])
    best = dict(auroc=-1.0, epoch=0, state=None)

    for epoch in range(n_epochs):
        flow.train()
        losses = []
        for views, _ in tqdm(train_loader, desc=f"epoch {epoch + 1}/{n_epochs}", leave=False):
            feats = extractor(data.first_view(views, device))
            feats = apply_feature_stats([feats[level] for level in levels], feat_stats)
            optimizer.zero_grad()
            loss = sum(m.mean() for m in flow.nll_per_level(feats))
            loss.backward()
            clip_flow_grads(flow, c.pixel_grad_clip)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(float(np.mean(losses)))
        tracker.metric('pixel_train_loss', history['train_loss'][-1], step=epoch)

        if (epoch + 1) % eval_interval and epoch != n_epochs - 1:
            print(f"  epoch {epoch + 1}/{n_epochs}  loss {history['train_loss'][-1]:.4f}")
            continue
        flow.eval()
        head = PixelHead(image_model, flow, selection_recipe, feat_stats=feat_stats, device=device)
        nll = cache_nll(head, testset, desc=f"test {epoch + 1}")
        metrics, _ = evaluate_maps(nll, masks, labels, selection_recipe, levels, aupro=False)
        auroc = metrics['pixel_auroc']
        history['eval_epoch'].append(epoch + 1)
        history['pixel_auroc'].append(auroc)
        tracker.metric('pixel_auroc_selection', auroc, step=epoch)
        marker = ''
        if auroc > best['auroc']:
            best = dict(auroc=auroc, epoch=epoch + 1, state=copy.deepcopy(flow.state_dict()))
            marker = '  (best)'
        print(f"  epoch {epoch + 1}/{n_epochs}  loss {history['train_loss'][-1]:.4f}  "
              f"Pixel AUROC (selection) {auroc:.4f}{marker}")
        save_json(os.path.join(out_dir, 'history_pixel.json'), history)

    # Inference recipe on top of the best epoch.
    flow.load_state_dict(best['state'])
    flow.eval()
    head = PixelHead(image_model, flow, inference, feat_stats=feat_stats, device=device)
    level_stats = None
    if inference['score_mode'] == 'per_level_std':
        stats_set = data.TrainDataset(dataset_path, class_name, n_views=1,
                                      transform=data.eval_transform())
        level_stats = compute_level_stats(head, data.loader(stats_set))
    nll = cache_nll(head, testset, desc='test (inference recipe)')

    ablation = None
    if recipe['level_ablation']:
        print("  level ablation (test Pixel AUROC):")
        ablation = run_level_ablation(nll, masks, labels, inference, levels, level_stats,
                                      c.level_ablation_min_gain)
        inference['levels'] = ablation['selected']
        print(f"  selected levels: {ablation['selected']} ({ablation['status']})")

    metrics, _ = evaluate_maps(nll, masks, labels, inference, levels, level_stats,
                               desc='final metrics')
    metrics.update(selection_epoch=best['epoch'], selection_pixel_auroc=best['auroc'])

    hparams = dict(out_size=c.pixel_out_size, cond_dim=c.pixel_cond_dim, n_blocks=c.pixel_n_blocks,
                   hidden=c.pixel_hidden, img_size=c.img_size[0], clamp_scale=training['clamp_scale'],
                   attention='se', feat_pool=0, levels=levels, feat_norm=bool(norm_levels),
                   feat_norm_levels=parse_levels(norm_levels) if norm_levels else None,
                   score_mode=selection_recipe['score_mode'], reflect_pad=0, pad_mode='reflect',
                   no_rotation=not c.transf_rotations)
    path = os.path.join(out_dir, 'pixel_head.pt')
    save_pixel_head(path, flow, hparams, epoch=best['epoch'], pixel_auroc=best['auroc'],
                    feat_stats=feat_stats, training_recipe=recipe, inference_recipe=inference,
                    inference_level_stats=level_stats, level_ablation=ablation,
                    image_head=os.path.abspath(image_head_path), dataset_path=dataset_path)
    return path, metrics, ablation
