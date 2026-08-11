"""Stage 2 classification backends.

``MockClassifier`` runs with zero deep-learning dependencies, so the server boots
and the full request path works without torch.

``ConvNeXtTinyClassifier`` is the real backend. Its architecture and eval transform
are faithful reconstructions of the training code
(``classification/datapreparepipeline/trainer_engine.build_convnext_tiny`` and
``dataset.get_eval_transforms``). Two guards keep it honest:

  * ``load_state_dict(strict=True)`` -- any architecture divergence fails loudly at
    load rather than silently mispredicting;
  * ``app/tests/test_convnext_parity.py`` -- reproduces the committed
    ``study_summary.json`` validation confusion matrix exactly, proving there is no
    train/serve skew in Stage 2.

torch / torchvision / albumentations are imported lazily inside the class so the
mock path (and merely importing this module) needs none of them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from app.models.base import BaseClassifier, ClassificationResult

# ImageNet normalization mirrors classification/datapreparepipeline/dataset.py.
# Kept here as the serving source of truth; the golden parity test proves equivalence.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _label(prob: float, threshold: float) -> str:
    # Strict ">" matches training's compute_metrics: preds = (probs > threshold).
    return "anemic" if prob > threshold else "non-anemic"


class MockClassifier(BaseClassifier):
    """Deterministic placeholder: same crop -> same probability, no real model."""

    name = "mock-classifier-v0"
    is_mock = True

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def predict(self, crop: Image.Image) -> ClassificationResult:
        digest = hashlib.sha256(crop.tobytes()).digest()
        prob = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        return ClassificationResult(
            probability=round(prob, 4),
            label=_label(prob, self._threshold),
            threshold=self._threshold,
            backend=self.name,
            is_mock=True,
        )


class ConvNeXtTinyClassifier(BaseClassifier):
    """Real Stage-2 backend: frozen ConvNeXt-Tiny + trained Dropout->Linear head.

    Decoupled from ``app.config`` on purpose (explicit args, not a Settings object)
    so it can be constructed from a test or a CLI without pulling in the web stack.
    """

    name = "convnext_tiny_palpebral_v2_clean"
    is_mock = False

    def __init__(
        self,
        weights_path: str | Path,
        *,
        input_size: int = 256,
        threshold: float = 0.5,
        device: str = "cpu",
        dropout_rate: float = 0.2,
    ) -> None:
        import albumentations as A
        import torch
        from albumentations.pytorch import ToTensorV2

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"ConvNeXt-Tiny weights not found at {weights_path}. Place the checkpoint "
                f"there, or set EYESDEFY_CLASSIFIER_BACKEND=mock."
            )

        self._torch = torch
        self._device = torch.device(device)
        self._threshold = threshold
        self._input_size = input_size

        self._model = self._build_model(dropout_rate)
        state = torch.load(weights_path, map_location=self._device)
        self._model.load_state_dict(state, strict=True)  # strict = architecture parity guard
        self._model.to(self._device).eval()

        # Eval transform: identical ops to dataset.get_eval_transforms(input_size).
        # (The 256x256 processed crops make Resize a no-op in training; on a
        # variable-size synthesized crop it does the real down/upsample.)
        self._transform = A.Compose(
            [
                A.Resize(input_size, input_size),
                A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def _build_model(dropout_rate: float):
        """Reconstruct trainer_engine.build_convnext_tiny exactly.

        ``weights=None`` because every weight (frozen backbone + trained head) comes
        from our checkpoint -- no ImageNet download at startup. The module structure
        is identical to the pretrained build, so strict state_dict loading matches.
        """
        import torch.nn as nn
        from torchvision import models

        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features  # 768
        model.classifier[2] = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))
        return model

    def predict(self, crop: Image.Image) -> ClassificationResult:
        torch = self._torch
        arr = np.asarray(crop.convert("RGB"))
        tensor = self._transform(image=arr)["image"].unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logit = self._model(tensor).reshape(-1)[0]
            prob = float(torch.sigmoid(logit).item())
        return ClassificationResult(
            probability=round(prob, 6),
            label=_label(prob, self._threshold),
            threshold=self._threshold,
            backend=self.name,
            is_mock=False,
        )
