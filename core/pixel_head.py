"""Pixel head: position-conditioned normalizing flow on frozen features.

Pipeline for one image
----------------------
1. :class:`FeatureExtractor` reads the activations of the frozen SEDifferNet at
   three levels (L1/L2/L3 = 64/192/256 channels) and resamples them to an
   ``out_size x out_size`` grid (96). With ``pad > 0`` the input is extended by
   reflection/replication before the backbone and the matching border is cropped.
2. :class:`PixelFlow` holds one flow per level; every grid position is a sample
   conditioned on a sinusoidal positional encoding. The output is the NLL per
   position and per level (:meth:`PixelFlow.nll_per_level`).
3. :meth:`PixelHead.nll` repeats step 2 on the mirrored views (TTA) and averages.
4. :func:`score_map` aggregates the selected levels into an ``img_size x img_size``
   map and applies the Gaussian; :func:`image_score_from_map` reduces the map to
   a single number.

Aggregation modes (``score_mode``)
----------------------------------
``raw``            sum of the per-level NLLs.
``per_level_std``  each level standardized by scalar mean/std estimated on the
                   training images (:func:`compute_level_stats`), then summed.
``cflow_original`` CFLOW-AD reference aggregation: per level, the per-channel
                   log-likelihood minus the cohort maximum, exponentiated; the
                   levels are summed and the map is ``global max - sum``. It
                   depends on the whole cohort, hence :class:`Cohort`.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch import nn

import config as c

LEVEL_CHANNELS = (64, 192, 256)
SCORE_MODES = ('raw', 'per_level_std', 'cflow_original')
TTA_MODES = {'none': ('id',), 'hflip': ('id', 'h'), 'flips': ('id', 'h', 'v', 'hv')}
PAD_MODES = ('reflect', 'replicate')

# Cumulative stride of the backbone down to the deepest level (stride-4 conv1 + two
# stride-2 max-pools). The padding must be a multiple of it so that the cropped
# grid stays registered with the mask.
BACKBONE_STRIDE = 16


def parse_levels(levels):
    """``"02"`` / ``"0,2"`` / ``[0, 2]`` / ``None`` -> sorted list of levels (``None`` = all)."""
    if levels is None:
        return [0, 1, 2]
    if isinstance(levels, str):
        levels = [int(ch) for ch in levels if ch.isdigit()]
    levels = sorted(set(int(level) for level in levels))
    if not levels or any(level not in (0, 1, 2) for level in levels):
        raise ValueError(f"invalid levels: {levels!r}; use a subset of 0, 1, 2")
    return levels


def levels_to_str(levels):
    return ''.join(str(level) for level in parse_levels(levels))


# ═══════════════════════════════════════════════════════════════════════════════
# Features
# ═══════════════════════════════════════════════════════════════════════════════

def snap_pad(pad, out_size, img_size, stride=BACKBONE_STRIDE):
    """Round the padding up to a value that keeps the grid aligned.

    The padding must be a multiple of the backbone stride and map to a whole
    number of cells of the ``out_size`` grid.
    """
    if pad <= 0:
        return 0
    cell_step = img_size // math.gcd(out_size, img_size)
    step = abs(stride * cell_step) // math.gcd(stride, cell_step)
    return max(step, int(round(pad / step)) * step)


class FeatureExtractor(nn.Module):
    """L1/L2/L3 activations of the frozen SEDifferNet on the ``out_size`` grid.

    Args:
        model: Trained SEDifferNet (only the backbone and the SE blocks are used).
        out_size: Side of the output grid.
        pad: Input padding in pixels (0 = no padding).
        pad_mode: ``'reflect'`` or ``'replicate'``.
    """

    def __init__(self, model, out_size=c.pixel_out_size, pad=0, pad_mode='reflect'):
        super().__init__()
        if pad_mode not in PAD_MODES:
            raise ValueError(f"unknown pad_mode {pad_mode!r}; use {PAD_MODES}")
        self.alexnet = model.alexnet
        # The extractor uses all four SE blocks, including simsa3, which the
        # SEDifferNet forward pass does not use. Every head was trained this way.
        self.se1, self.se2, self.se3, self.se4 = model.simsa1, model.simsa2, model.simsa3, model.simsa4
        self.out_size, self.pad, self.pad_mode = out_size, pad, pad_mode
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        """``(B, 3, H, W)`` -> list with the three levels, each ``(B, C, out, out)``."""
        target, crop = self.out_size, 0
        if self.pad > 0:
            pad = snap_pad(self.pad, self.out_size, x.shape[-2])
            crop = pad * self.out_size // x.shape[-2]
            target = self.out_size + 2 * crop
            x = F.pad(x, (pad,) * 4, mode=self.pad_mode)

        f = self.alexnet.features
        l1 = self.se1(f[1](f[0](x)))
        l2 = self.se2(f[4](f[3](f[2](l1))))
        l3 = self.se3(f[7](f[6](f[5](l2))))
        l3 = self.se4(f[11](f[10](f[9](f[8](l3)))))

        levels = [F.interpolate(t, size=(target, target), mode='bilinear', align_corners=False)
                  for t in (l1, l2, l3)]
        if crop > 0:
            levels = [t[:, :, crop:crop + self.out_size, crop:crop + self.out_size] for t in levels]
        return levels


def compute_feature_stats(extractor, data_loader, device, norm_levels):
    """Per-channel mean/std of the levels in ``norm_levels`` over the training images.

    Returns:
        List with one item per level (0, 1, 2): ``(mean, std)`` with shape ``(C,)``,
        or ``None`` for levels that are not standardized.
    """
    norm_levels = set(parse_levels(norm_levels))
    sums, sqs, count = [None] * 3, [None] * 3, 0
    for images, _ in data_loader:
        feats = extractor(images[:, 0].to(device))
        for i, f in enumerate(feats):
            if i not in norm_levels:
                continue
            s, q = f.double().sum(dim=(0, 2, 3)), (f.double() ** 2).sum(dim=(0, 2, 3))
            sums[i] = s if sums[i] is None else sums[i] + s
            sqs[i] = q if sqs[i] is None else sqs[i] + q
        count += feats[0].shape[0] * feats[0].shape[2] * feats[0].shape[3]

    stats = []
    for s, q in zip(sums, sqs):
        if s is None:
            stats.append(None)
            continue
        mean = s / count
        std = (q / count - mean ** 2).clamp_min(0.0).sqrt()
        stats.append((mean.float().cpu(), std.float().cpu()))
    return stats


def apply_feature_stats(feats, feat_stats, eps=1e-6):
    """Standardize each level per channel; ``None`` leaves the level unchanged."""
    if feat_stats is None:
        return feats
    out = []
    for f, st in zip(feats, feat_stats):
        if st is None:
            out.append(f)
        else:
            mu, sd = (t.to(f.device).view(1, -1, 1, 1) for t in st)
            out.append((f - mu) / (sd + eps))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional flow
# ═══════════════════════════════════════════════════════════════════════════════

def positional_encoding(h, w, dim, device=None):
    """2D sinusoidal positional encoding, ``(h * w, dim)``; ``dim`` must be a multiple of 4."""
    assert dim % 4 == 0
    quarter = dim // 4
    div = torch.exp(torch.arange(0, quarter, device=device).float() * (-math.log(10000.0) / quarter))
    yy = torch.arange(h, device=device).float().unsqueeze(1) * div.unsqueeze(0)
    xx = torch.arange(w, device=device).float().unsqueeze(1) * div.unsqueeze(0)
    pe_y = torch.cat([yy.sin(), yy.cos()], dim=1).unsqueeze(1).expand(h, w, dim // 2)
    pe_x = torch.cat([xx.sin(), xx.cos()], dim=1).unsqueeze(0).expand(h, w, dim // 2)
    return torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, dim)


class CondCouplingBlock(nn.Module):
    """Conditional affine coupling; log-scale bounded by ``tanh(s) * clamp_scale``."""

    def __init__(self, channels, cond_dim, hidden, clamp_scale):
        super().__init__()
        self.c_split = channels // 2
        self.clamp_scale = clamp_scale
        self.net = nn.Sequential(
            nn.Linear(channels - self.c_split + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * self.c_split),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, cond):
        x1, x2 = x[:, :self.c_split], x[:, self.c_split:]
        s, t = self.net(torch.cat([x2, cond], dim=1)).chunk(2, dim=1)
        s = torch.tanh(s) * self.clamp_scale
        return torch.cat([x1 * torch.exp(s) + t, x2], dim=1), s.sum(dim=1)


class CondFlow(nn.Module):
    """Stack of couplings with a fixed channel rotation between blocks."""

    def __init__(self, channels, cond_dim, n_blocks, hidden, clamp_scale):
        super().__init__()
        self.blocks = nn.ModuleList([CondCouplingBlock(channels, cond_dim, hidden, clamp_scale)
                                     for _ in range(n_blocks)])
        self.register_buffer('perm_idx', torch.tensor(
            [(i + channels // 2) % channels for i in range(channels)]))

    def forward(self, x, cond):
        log_det = torch.zeros(x.size(0), device=x.device)
        for block in self.blocks:
            x, ld = block(x, cond)
            log_det = log_det + ld
            x = x.index_select(1, self.perm_idx)
        return x, log_det


class PixelFlow(nn.Module):
    """One conditional flow per trained level (weights stored as ``cflow_state_dict``)."""

    def __init__(self, levels, cond_dim, n_blocks, hidden, clamp_scale):
        super().__init__()
        self.levels = parse_levels(levels)
        self.cond_dim = cond_dim
        self.flows = nn.ModuleList([
            CondFlow(LEVEL_CHANNELS[level], cond_dim, n_blocks, hidden, clamp_scale)
            for level in self.levels])

    def nll_per_level(self, feats, chunk_size=4096):
        """Per-position NLL of each trained level.

        Args:
            feats: Features of the trained levels, each ``(B, C, H, W)``.
            chunk_size: Positions per step (bounds VRAM; does not change the result).

        Returns:
            List of ``(B, H, W)``.
        """
        maps = []
        for feat, flow in zip(feats, self.flows):
            B, C, H, W = feat.shape
            pe = positional_encoding(H, W, self.cond_dim, feat.device)
            cond = pe.unsqueeze(0).expand(B, -1, -1).reshape(B * H * W, self.cond_dim)
            x = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            chunks = []
            for i in range(0, x.shape[0], chunk_size):
                z, log_det = flow(x[i:i + chunk_size], cond[i:i + chunk_size])
                log_prob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * C * math.log(2 * math.pi)
                chunks.append(-(log_prob + log_det))
            maps.append(torch.cat(chunks).view(B, H, W))
        return maps


def clip_flow_grads(pixel_flow, max_norm):
    """Clip the gradient of each level separately (a global norm would favour the widest level)."""
    for flow in pixel_flow.flows:
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm)


def flip(x, mode):
    """TTA mirroring on the last two dimensions; every mode is its own inverse."""
    if mode == 'h':
        return x.flip(-1)
    if mode == 'v':
        return x.flip(-2)
    if mode == 'hv':
        return x.flip(-1).flip(-2)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# Complete head (extractor + flow + statistics + recipe)
# ═══════════════════════════════════════════════════════════════════════════════

class PixelHead:
    """Pixel head ready for inference.

    Bundles the trained flow, the standardization statistics and the inference
    recipe. ``nll`` returns the per-level NLL (TTA average); turning it into a map
    is done by :func:`score_map`, which needs the cohort for ``cflow_original``.
    """

    def __init__(self, image_model, flow, recipe, feat_stats=None, level_stats=None, device='cpu'):
        self.flow = flow.to(device).eval()
        self.recipe = dict(recipe)
        self.feat_stats = feat_stats
        self.level_stats = level_stats
        self.device = device
        self.trained_levels = flow.levels
        self.extractor = FeatureExtractor(image_model, pad=self.recipe['pad'],
                                          pad_mode=self.recipe['pad_mode']).to(device).eval()

    def trained_features(self, images):
        feats = self.extractor(images)
        feats = [feats[level] for level in self.trained_levels]
        return apply_feature_stats(feats, self.feat_stats)

    @torch.no_grad()
    def nll(self, images, tta=None):
        """NLL per trained level, averaged over the TTA views: ``(B, L, out, out)`` in float32."""
        modes = TTA_MODES[tta or self.recipe['tta']]
        total = None
        for mode in modes:
            maps = self.flow.nll_per_level(self.trained_features(flip(images, mode)))
            maps = torch.stack([flip(m, mode) for m in maps], dim=1)
            total = maps if total is None else total + maps
        return (total / len(modes)).float().cpu()


@torch.no_grad()
def compute_level_stats(head, data_loader):
    """Scalar mean/std of the NLL of each trained level over the training images.

    Uses the extractor of the inference recipe (same padding), without TTA. This
    is the standardization of the ``per_level_std`` mode.
    """
    sums = sqs = None
    count = 0
    for images, _ in data_loader:
        nll = head.nll(images[:, 0].to(head.device), tta='none').double()
        level_sums, level_sqs = nll.sum(dim=(0, 2, 3)), (nll ** 2).sum(dim=(0, 2, 3))
        sums = level_sums if sums is None else sums + level_sums
        sqs = level_sqs if sqs is None else sqs + level_sqs
        count += nll.shape[0] * nll.shape[2] * nll.shape[3]
    mean = sums / count
    std = (sqs / count - mean ** 2).clamp_min(0.0).sqrt()
    return [(float(m), float(s)) for m, s in zip(mean, std)]


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation into a map
# ═══════════════════════════════════════════════════════════════════════════════

class Cohort:
    """Cohort constants required by ``cflow_original`` (per-level and global maxima).

    ``raw`` and ``per_level_std`` have no dependency between images, so the cohort
    stays empty for them.
    """

    def __init__(self, level_max=None, global_max=None):
        self.level_max = level_max
        self.global_max = global_max

    @classmethod
    def from_cache(cls, nll_cache, recipe, trained_levels):
        if recipe['score_mode'] != 'cflow_original':
            return cls()
        index = _level_index(recipe['levels'], trained_levels)
        channels = [LEVEL_CHANNELS[trained_levels[i]] for i in index]
        level_max = [max(float((-torch.as_tensor(nll[i]).double() / ch).max()) for nll in nll_cache)
                     for i, ch in zip(index, channels)]
        cohort = cls(level_max=level_max)
        cohort.global_max = max(float(_cflow_probability(nll, recipe, trained_levels, cohort).max())
                                for nll in nll_cache)
        return cohort


def _level_index(levels, trained_levels):
    wanted = parse_levels(levels)
    missing = [level for level in wanted if level not in trained_levels]
    if missing:
        raise ValueError(f"the recipe asks for levels {missing}, but the head was only trained "
                         f"on {trained_levels}")
    return [trained_levels.index(level) for level in wanted]


def _upsample(t, align_corners):
    return F.interpolate(t.unsqueeze(1), size=c.img_size, mode='bilinear',
                         align_corners=align_corners).squeeze(1)


def _cflow_probability(nll, recipe, trained_levels, cohort):
    nll = torch.as_tensor(nll).unsqueeze(0)
    total = 0
    for k, i in enumerate(_level_index(recipe['levels'], trained_levels)):
        log_prob = -nll[:, i].double() / LEVEL_CHANNELS[trained_levels[i]]
        total = total + _upsample(torch.exp(log_prob - cohort.level_max[k]), recipe['align_corners'])
    return total[0]


def score_map(nll, recipe, trained_levels, level_stats=None, cohort=None):
    """Anomaly map of one image, already smoothed.

    Args:
        nll: ``(L, out, out)`` NLL of the trained levels (output of :meth:`PixelHead.nll`).
        recipe: Inference recipe (``score_mode``, ``levels``, ``align_corners``, ``sigma``).
        trained_levels: Levels the head was trained on.
        level_stats: ``[(mean, std)]`` per trained level, for ``per_level_std``.
        cohort: :class:`Cohort` for ``cflow_original``.

    Returns:
        ``(H, W)`` numpy array; higher = more anomalous.
    """
    mode = recipe['score_mode']
    if mode not in SCORE_MODES:
        raise ValueError(f"unknown score_mode {mode!r}; use {SCORE_MODES}")
    if mode == 'cflow_original':
        if cohort is None or cohort.global_max is None:
            raise ValueError("cflow_original needs the cohort (Cohort.from_cache)")
        result = cohort.global_max - _cflow_probability(nll, recipe, trained_levels, cohort)
    else:
        nll = torch.as_tensor(nll).unsqueeze(0)
        result = 0
        for i in _level_index(recipe['levels'], trained_levels):
            m = nll[:, i]
            if mode == 'per_level_std':
                if level_stats is None:
                    raise ValueError("per_level_std needs level_stats")
                mu, sd = level_stats[i]
                m = (m - mu) / (sd + 1e-8)
            result = result + _upsample(m, recipe['align_corners'])
        result = result[0]
    result = result.numpy()
    sigma = recipe['sigma']
    return gaussian_filter(result, sigma=sigma) if sigma > 0 else result


def image_score_from_map(amap, mode='max', top_percent=1.0):
    """Reduce an ``(H, W)`` map to one score: maximum or mean of the top ``top_percent`` %."""
    flat = np.asarray(amap).reshape(-1)
    if mode == 'max':
        return float(flat.max())
    if mode != 'topk':
        raise ValueError(f"unknown image_score {mode!r}")
    k = max(1, int(flat.size * top_percent / 100.0))
    return float(-np.partition(-flat, k - 1)[:k].mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

def build_flow(hparams):
    return PixelFlow(levels=hparams.get('levels'), cond_dim=hparams.get('cond_dim', c.pixel_cond_dim),
                     n_blocks=hparams.get('n_blocks', c.pixel_n_blocks),
                     hidden=hparams.get('hidden', c.pixel_hidden),
                     clamp_scale=hparams.get('clamp_scale', 0.5))


def save_pixel_head(path, flow, hparams, **extra):
    """Save the flow with everything inference needs (hyperparameters, recipe, statistics)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'cflow_state_dict': flow.state_dict(), 'cflow_hparams': dict(hparams), **extra}, path)


def load_pixel_head(path, image_model, device, recipe_override=None):
    """Load a pixel head.

    Also reads the project's older checkpoints (``best_pixel_auroc.pt``), which
    have the same keys. The inference recipe stored in the checkpoint is used
    unless ``recipe_override`` is given.

    Returns:
        ``(PixelHead, checkpoint)``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"pixel-head checkpoint not found: {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    hparams = {k: v for k, v in (ckpt.get('cflow_hparams') or {}).items() if v is not None}
    _check_supported(hparams)
    flow = build_flow(hparams)
    flow.load_state_dict(ckpt['cflow_state_dict'])

    recipe = dict(c.PIXEL_INFERENCE_DEFAULTS)
    recipe.update(recipe_override or ckpt.get('inference_recipe') or {})
    _check_recipe(recipe)
    feat_stats = ckpt.get('feat_stats')
    if feat_stats is not None and len(feat_stats) != len(flow.levels):
        raise ValueError("feat_stats does not match the trained levels")
    head = PixelHead(image_model, flow, recipe, feat_stats=feat_stats,
                     level_stats=ckpt.get('inference_level_stats'), device=device)
    return head, ckpt


def _check_supported(hparams):
    """Reject checkpoints of experimental variants that this lean code does not reproduce."""
    expected = dict(out_size=c.pixel_out_size, img_size=c.img_size[0], attention='se', feat_pool=0)
    for key, value in expected.items():
        if key in hparams and hparams[key] != value:
            raise ValueError(f"checkpoint has {key}={hparams[key]!r}; this code only supports {value!r}")


def _check_recipe(recipe):
    if recipe['score_mode'] not in SCORE_MODES:
        raise ValueError(f"score_mode {recipe['score_mode']!r} is not supported; use {SCORE_MODES}")
    if recipe['tta'] not in TTA_MODES:
        raise ValueError(f"tta {recipe['tta']!r} is not supported; use {tuple(TTA_MODES)}")
    if recipe.get('margin', 0) not in (0, None) or recipe.get('pos_calib', 'none') not in ('none', None):
        raise ValueError("recipes with a margin or positional calibration are not supported")
