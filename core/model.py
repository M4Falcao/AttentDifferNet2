"""Image head: SEDifferNet.

DifferNet (Rudolph et al., WACV 2021) with Squeeze-and-Excitation blocks inside
AlexNet. Features from three image scales are globally average-pooled,
concatenated and mapped by a normalizing flow to a standard Gaussian. The
anomaly score is the mean squared latent norm over the views of an image.

Only the flow (``model.nf``) is trained; AlexNet (ImageNet) and the SE blocks
keep their seeded initialization. Therefore the activations read by the pixel
head are the same for any encoder trained with the same seed.
"""

import os

import torch
import torch.nn.functional as F
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.SEAttention import SEAttention
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet

import config as c
from core.freia_funcs import (F_fully_connected, InputNode, Node, OutputNode, ReversibleGraphNet,
                              glow_coupling_layer, permute_layer)


def nf_head(input_dim=c.n_feat):
    """Image-head flow: ``n_coupling_blocks`` pairs of (fixed permutation, GLOW coupling)."""
    nodes = [InputNode(input_dim, name='input')]
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=f'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': c.fc_internal, 'dropout': c.dropout}},
                          name=f'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    return ReversibleGraphNet(nodes)


class SEDifferNet(nn.Module):
    """DifferNet with SE attention after AlexNet stages 1, 2 and 4.

    The ``cbam*`` and ``simsa3`` blocks are not used in ``forward``. They are still
    built because (1) existing checkpoints contain them and (2) the construction
    order defines how the random generator is consumed, and therefore the
    initialization of the SE blocks with seed 42.
    """

    def __init__(self):
        super().__init__()
        self.alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.cbam1 = CBAMBlock(channel=64, reduction=16, kernel_size=49)
        self.cbam2 = CBAMBlock(channel=192, reduction=16, kernel_size=49)
        self.cbam3 = CBAMBlock(channel=384, reduction=16, kernel_size=49)
        self.cbam4 = CBAMBlock(channel=256, reduction=16, kernel_size=49)
        self.simsa1 = SEAttention(channel=64, reduction=2)
        self.simsa2 = SEAttention(channel=192, reduction=2)
        self.simsa3 = SEAttention(channel=384, reduction=2)
        self.simsa4 = SEAttention(channel=256, reduction=2)
        self.nf = nf_head()

    def features(self, x):
        """Feature vector: global average of the last stage at ``n_scales`` scales."""
        pooled = []
        for s in range(c.n_scales):
            x_scaled = F.interpolate(x, size=c.img_size[0] // (2 ** s)) if s > 0 else x
            f = self.alexnet.features
            h = self.simsa1(f[1](f[0](x_scaled)))
            h = self.simsa2(f[4](f[3](f[2](h))))
            h = f[11](f[10](f[9](f[8](f[7](f[6](f[5](h)))))))
            h = f[12](self.simsa4(h))
            pooled.append(torch.mean(h, dim=(2, 3)))
        return torch.cat(pooled, dim=1)

    def forward(self, x):
        return self.nf(self.features(x))


def nll_loss(z, log_jac_det):
    """Flow loss (eq. 4 of DifferNet), normalized by the latent dimension."""
    return torch.mean(0.5 * torch.sum(z ** 2, dim=1) - log_jac_det) / z.shape[1]


@torch.no_grad()
def image_anomaly_score(model, views):
    """Anomaly score of one image: mean of ``z**2`` over all its views.

    Args:
        model: SEDifferNet in ``eval`` mode.
        views: ``(V, 3, H, W)`` with the views of a single image.
    """
    return float(torch.mean(model(views) ** 2).item())


def load_image_head(path, device):
    """Load a SEDifferNet saved as a checkpoint, a ``state_dict`` or a whole module."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"image-head checkpoint not found: {path}")
    content = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(content, nn.Module):
        state = content.state_dict()
    elif isinstance(content, dict) and 'model_state_dict' in content:
        state = content['model_state_dict']
    else:
        state = content
    model = SEDifferNet()
    model.load_state_dict(state)
    return model.to(device).eval()


def save_image_head(path, model, **metadata):
    """Save the SEDifferNet weights with metadata (epoch, AUROC, ...)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), **metadata}, path)
