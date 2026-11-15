"""Single project configuration.

Everything that defines an experiment lives in this file: data, hyperparameters of
both heads, pixel-head recipes and output paths. The scripts (``train.py``,
``infer.py``, ``evaluate.py``) only read these values; command-line arguments
override some of them for a single run without changing the file. No module
writes to this module at runtime.

Terminology
-----------
- **Image head**: SEDifferNet (AlexNet + Squeeze-and-Excitation + normalizing
  flow). Decides whether an image is anomalous.
- **Pixel head**: position-conditioned normalizing flow trained on the frozen
  activations of the image head. Produces the anomaly map.
- **Levels**: backbone activations L1/L2/L3, indexed ``0/1/2``. A string such as
  ``"02"`` means L1+L3; ``"012"`` means all levels.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime
# ═══════════════════════════════════════════════════════════════════════════════

device = "cuda"           # falls back to "cpu" automatically when no GPU is available
seed = 42
num_workers = 0

# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════
# Expected layout (MVTec): <dataset_path>/<class>/train/good,
# <class>/test/<good|defect>, <class>/ground_truth/<defect>.
#
# To use the background-free version, point to the folder produced by
# ../yolo-segmentation/ (e.g. .../insplad-seg-no-back-treino).
# Relative paths are resolved from the folder of this file.

dataset_path = os.path.join(_HERE, "..", "..", "..", "anomaly-detection-dataset",
                            "insplad-seg", "insplad-seg")

classes = [
    "glass-insulator",
    "lightning-rod-suspension",
    "polymer-insulator-upper-shackle",
    "vari-grip",
    "yoke-suspension",
]

img_size = (448, 448)
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Training augmentation (both heads use the same training transform).
transf_rotations = True
transf_brightness = 0.0
transf_contrast = 0.0
transf_saturation = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# Image head (SEDifferNet)
# ═══════════════════════════════════════════════════════════════════════════════

n_scales = 3               # image scales: img_size, img_size/2, img_size/4
n_feat = 256 * n_scales    # feature-vector width (fixed by AlexNet)
n_coupling_blocks = 8
clamp_alpha = 3
fc_internal = 2048
dropout = 0.0
lr_init = 2e-4

n_transforms = 4           # random views per image during training
n_transforms_test = 64     # fixed rotations per image for the test score
batch_size = 1             # effective batch = batch_size * n_transforms

pre_epochs = 30
meta_epochs = 170          # the epoch is selected by test Image AUROC (64 fixed rotations)

# ═══════════════════════════════════════════════════════════════════════════════
# Pixel head (position-conditioned normalizing flow)
# ═══════════════════════════════════════════════════════════════════════════════

pixel_out_size = 96        # grid of the features and of the positional encoding
pixel_cond_dim = 64        # width of the positional encoding
pixel_n_blocks = 8
pixel_hidden = 256
pixel_lr = 2e-4
pixel_warmup_epochs = 5
pixel_grad_clip = 1.0

# How the pixel-head epoch is selected: test Pixel AUROC with the "raw" score,
# all levels, no padding or TTA, and a Gaussian with sigma 4.
pixel_selection = dict(score_mode="raw", sigma=4.0)

# Default recipe. Options: "best_per_class", "general", "baseline".
pixel_recipe = "best_per_class"

# The recipes below were measured on the original INSPLAD-seg (with background).
# They only apply to that dataset; every other case uses the fallback recipe.
planned_dataset = "insplad-seg"   # folder name of dataset_path

# Starting values of every recipe. Each recipe overrides only what changes.
#   training.levels            levels the head is trained on
#   training.feat_norm_levels  levels standardized per channel before the flow (None = none)
#   training.clamp_scale       bound on the log-scale of the affine couplings
#   inference.pad / pad_mode   input padding before the backbone (pixels)
#   inference.tta              "none", "hflip" or "flips" (4 mirrored views)
#   inference.score_mode       "raw", "per_level_std" or "cflow_original"
#   inference.levels           levels summed into the final map
#   inference.align_corners    mode of the bilinear upsampling of the map
#   inference.sigma            Gaussian applied to the final map
#   inference.image_score      "max" or "topk" (mean of the top topk_percent %)
PIXEL_TRAINING_DEFAULTS = dict(levels="012", feat_norm_levels=None, clamp_scale=0.5)
PIXEL_INFERENCE_DEFAULTS = dict(pad=0, pad_mode="reflect", tta="none", score_mode="raw",
                                levels="012", align_corners=False, sigma=4.0,
                                image_score="max", topk_percent=1.0)

# Pixel-head epochs and evaluation interval, per class.
PIXEL_EPOCHS = {
    "glass-insulator": (40, 5),
    "lightning-rod-suspension": (40, 5),
    "polymer-insulator-upper-shackle": (30, 3),
    "vari-grip": (40, 5),
    "yoke-suspension": (12, 2),
}

PIXEL_RECIPES = {
    # Legacy inference: no padding, no TTA, plain sum of the NLLs.
    "baseline": dict(training={}, inference={}, per_class={}),

    # One inference recipe shared by all classes; only the levels change.
    "general": dict(
        training={},
        inference=dict(pad=112, tta="flips", score_mode="per_level_std",
                       align_corners=True, sigma=8.0),
        per_class={
            "glass-insulator": dict(inference=dict(levels="02")),
            "lightning-rod-suspension": dict(inference=dict(levels="012")),
            "polymer-insulator-upper-shackle": dict(inference=dict(levels="12")),
            "vari-grip": dict(inference=dict(levels="0")),
            "yoke-suspension": dict(inference=dict(levels="012")),
        },
    ),

    # Winning training variant and inference recipe of each class.
    "best_per_class": dict(
        training={},
        inference=dict(pad=112, image_score="topk"),
        per_class={
            "glass-insulator": dict(
                training=dict(feat_norm_levels="012"),
                inference=dict(pad_mode="reflect", tta="flips", score_mode="per_level_std",
                               levels="02", align_corners=False, sigma=2.0)),
            "lightning-rod-suspension": dict(
                training=dict(clamp_scale=1.9),
                inference=dict(pad_mode="replicate", tta="flips", score_mode="cflow_original",
                               levels="012", align_corners=True, sigma=8.0)),
            "polymer-insulator-upper-shackle": dict(
                training=dict(feat_norm_levels="012", clamp_scale=1.9),
                inference=dict(pad_mode="replicate", tta="hflip", score_mode="per_level_std",
                               levels="012", align_corners=False, sigma=4.0, image_score="max")),
            "vari-grip": dict(
                training=dict(feat_norm_levels="012", clamp_scale=1.9),
                inference=dict(pad_mode="reflect", tta="hflip", score_mode="cflow_original",
                               levels="0", align_corners=True, sigma=8.0)),
            "yoke-suspension": dict(
                training=dict(feat_norm_levels="0"),
                inference=dict(pad_mode="reflect", tta="flips", score_mode="per_level_std",
                               levels="02", align_corners=True, sigma=12.0)),
        },
    ),
}

# Fallback: dataset or class without a planned recipe. Uses the "general" recipe
# with all levels and, at the end of training, a quick ablation decides which
# levels enter the map (the result is stored in the head checkpoint).
FALLBACK_RECIPE = dict(
    training={},
    inference=dict(pad=112, tta="flips", score_mode="per_level_std", levels="012",
                   align_corners=True, sigma=8.0),
)
FALLBACK_EPOCHS = (40, 5)

# Level ablation: "auto" (fallback only), "always" or "never".
level_ablation = "auto"
# A smaller subset only replaces "012" if it gains at least this much Pixel AUROC.
level_ablation_min_gain = 0.005

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

# Each training run creates <output_dir>/<timestamp>/<class>/{image_head.pt, pixel_head.pt, ...}.
output_dir = os.path.join(_HERE, "runs")

# Already trained image heads, per class. When a class has a path, training skips
# the image head and only trains the pixel head on top of it.
image_head_checkpoints = {}

# Model folder used by infer.py and evaluate.py (layout of output_dir/<timestamp>).
model_dir = None
infer_output_dir = os.path.join(_HERE, "inference")
eval_output_dir = os.path.join(_HERE, "evaluation")

# ═══════════════════════════════════════════════════════════════════════════════
# MLflow (optional)
# ═══════════════════════════════════════════════════════════════════════════════

use_mlflow = False
mlflow_tracking_uri = "file:" + os.path.join(_HERE, "mlruns")
mlflow_experiment_name = "SEDifferNet"
