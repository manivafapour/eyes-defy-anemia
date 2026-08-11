"""Backend factory: turn settings into concrete model instances.

Centralizing construction here keeps ``main.py``'s lifespan handler tiny and makes
"which backend is active" a single, testable decision.
"""
from __future__ import annotations

from app.config import Settings
from app.models.base import BaseClassifier, BaseSegmenter
from app.models.classification import ConvNeXtTinyClassifier, MockClassifier
from app.models.segmentation import MockSegmenter


def build_segmenter(settings: Settings) -> BaseSegmenter:
    backend = settings.segmenter_backend.lower()
    if backend == "mock":
        return MockSegmenter()
    # if backend == "aligned_unet":
    #     return AlignedUNetSegmenter(settings)   # <- Stage 1 swap-in point
    raise ValueError(f"Unknown segmenter backend: {settings.segmenter_backend!r}")


def build_classifier(settings: Settings) -> BaseClassifier:
    backend = settings.classifier_backend.lower()
    if backend == "mock":
        return MockClassifier(threshold=settings.classifier_threshold)
    if backend == "convnext_tiny":
        return ConvNeXtTinyClassifier(
            weights_path=settings.weights_dir / settings.classifier_weights,
            input_size=settings.classifier_input_size,
            threshold=settings.classifier_threshold,
            device=settings.device,
        )
    raise ValueError(f"Unknown classifier backend: {settings.classifier_backend!r}")
