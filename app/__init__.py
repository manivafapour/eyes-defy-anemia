"""Eyes-Defy-Anemia serving application (MLOps / deployment phase).

Two-stage inference platform:

    Stage 1 (segmentation)   raw eye photo    -> conjunctiva mask   [MOCKED]
    Stage 2 (classification) conjunctiva crop -> P(anemic)          [ConvNeXt-Tiny]

The package is layered so the framework-agnostic inference core (``app.core``)
has no dependency on FastAPI, and the model backends (``app.models``) sit behind
abstract interfaces. That lets the mock Stage 1 be swapped for the real aligned
U-Net (and the mock Stage 2 for ConvNeXt-Tiny) without touching pipeline or API
code -- only a config flag changes.
"""
from __future__ import annotations

__version__ = "0.1.0"
