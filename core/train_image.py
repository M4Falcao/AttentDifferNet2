"""Image-head training (SEDifferNet).

Only the normalizing flow is optimized. After ``pre_epochs`` warm-up epochs, each
of the ``meta_epochs`` epochs ends with the test Image AUROC (64 fixed rotations
per image, the same score used in evaluation) and the best model is saved.
"""

import os

import numpy as np
import torch
from tqdm import tqdm

import config as c
from core import data
from core.evaluation import image_head_scores
from core.metrics import image_metrics
from core.model import SEDifferNet, nll_loss, save_image_head
from core.utils import save_json


def _train_epoch(model, optimizer, train_loader, device, desc):
    model.train()
    losses = []
    for views, _ in tqdm(train_loader, desc=desc, leave=False):
        optimizer.zero_grad()
        z = model(data.all_views(views, device))
        loss = nll_loss(z, model.nf.jacobian(run_forward=False))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def train_image_head(class_name, dataset_path, out_dir, device, tracker, epochs=None):
    """Train the SEDifferNet of one class.

    Args:
        class_name: Class.
        dataset_path: Dataset root.
        out_dir: Class folder of the run; receives ``image_head.pt`` and ``history_image.json``.
        device: Torch device.
        tracker: :class:`core.tracking.Tracker`.
        epochs: ``(pre_epochs, meta_epochs)`` for quick tests (default: ``config``).

    Returns:
        ``(path of image_head.pt, best Image AUROC)``.
    """
    pre_epochs, meta_epochs = epochs or (c.pre_epochs, c.meta_epochs)
    trainset = data.TrainDataset(dataset_path, class_name, n_views=c.n_transforms)
    testset = data.TestDataset(dataset_path, class_name, n_views=c.n_transforms_test)
    train_loader = data.loader(trainset, shuffle=True)
    print(f"  image head: {len(trainset)} training images, {len(testset)} test images")

    model = SEDifferNet()
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8),
                                 eps=1e-04, weight_decay=1e-5)
    model.to(device)

    for epoch in range(pre_epochs):
        loss = _train_epoch(model, optimizer, train_loader, device,
                            f"pre-epoch {epoch + 1}/{pre_epochs}")
        tracker.metric('image_pre_train_loss', loss, step=epoch)
        print(f"  pre-epoch {epoch + 1}/{pre_epochs}  loss {loss:.4f}")

    path = os.path.join(out_dir, 'image_head.pt')
    history = dict(epoch=[], train_loss=[], image_auroc=[])
    best_auroc = -1.0
    for epoch in range(meta_epochs):
        loss = _train_epoch(model, optimizer, train_loader, device, f"epoch {epoch + 1}/{meta_epochs}")
        model.eval()
        scores = image_head_scores(model, testset, device, desc=f"test {epoch + 1}")
        auroc = image_metrics(testset.labels, scores, prefix='').get('auroc', float('nan'))

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(loss)
        history['image_auroc'].append(auroc)
        save_json(os.path.join(out_dir, 'history_image.json'), history)
        tracker.metric('image_train_loss', loss, step=epoch)
        tracker.metric('image_auroc', auroc, step=epoch)

        marker = ''
        if auroc > best_auroc:
            best_auroc = auroc
            save_image_head(path, model, epoch=epoch + 1, image_auroc=auroc, class_name=class_name,
                            dataset_path=dataset_path)
            marker = '  (best)'
        print(f"  epoch {epoch + 1}/{meta_epochs}  loss {loss:.4f}  Image AUROC {auroc:.4f}{marker}")

    return path, best_auroc
