"""Abstract model interfaces + shared result types.

Framework-agnostic: no FastAPI, no I/O. Defines the contract every segmentation /
classification backend must satisfy, so the inference pipeline is written once
against the interface and the concrete model (mock now, real ConvNeXt-Tiny /
aligned U-Net later) is swapped in without touching pipeline code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray          # HxW, non-zero = conjunctiva tissue (image frame)
    coverage: float           # fraction of the frame flagged as tissue, [0, 1]
    backend: str              # identifier of the model that produced it
    is_mock: bool = False


@dataclass(frozen=True)
class ClassificationResult:
    probability: float        # P(anemic), [0, 1]
    label: str                # "anemic" | "non-anemic"
    threshold: float
    backend: str
    is_mock: bool = False


class BaseSegmenter(ABC):
    """Stage 1 contract: raw eye image -> tissue mask in the image's pixel frame."""

    name: str = "base-segmenter"
    is_mock: bool = False

    @abstractmethod
    def segment(self, image: Image.Image) -> SegmentationResult:
        ...


class BaseClassifier(ABC):
    """Stage 2 contract: synthesized conjunctiva crop -> P(anemic).

    The backend owns its own final preprocessing (resize + ImageNet normalize +
    to-tensor), because those steps belong to the model, mirroring how the training
    ``Dataset`` applied them.
    """

    name: str = "base-classifier"
    is_mock: bool = False

    @abstractmethod
    def predict(self, crop: Image.Image) -> ClassificationResult:
        ...
