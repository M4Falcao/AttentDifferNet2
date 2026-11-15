"""Project library.

- :mod:`core.data`           datasets and transforms
- :mod:`core.model`          image head (SEDifferNet)
- :mod:`core.pixel_head`     pixel head (position-conditioned flow)
- :mod:`core.recipes`        resolution of the pixel-head recipe
- :mod:`core.train_image`    image-head training
- :mod:`core.train_pixel`    pixel-head training
- :mod:`core.level_ablation` quick level ablation (fallback recipe)
- :mod:`core.evaluation`     test-set evaluation from the NLL cache
- :mod:`core.metrics`        AUROC, AP and AUPRO
- :mod:`core.inference`      complete detector (both heads)
- :mod:`core.tracking`       optional MLflow logging
- :mod:`core.utils`          device, seed, JSON and model paths
- :mod:`core.freia_funcs`    normalizing-flow primitives (vendored FrEIA)
"""
