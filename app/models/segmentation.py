"""Stage 1 segmentation backends.

The real model (an aligned raw-photo U-Net) has NOT been trained yet, so a mock is
used to exercise the full pipeline end-to-end. The mock is *honest*: it does not
look at the image, it returns a fixed, anatomically-plausible lower-eyelid region so
downstream code has a real mask to work with.

Swap-in point: implement ``AlignedUNetSegmenter(BaseSegmenter)`` here, load the
trained checkpoint, run inference (logits -> sigmoid -> threshold, then upsample the
mask back to the raw image frame), and select it via ``EYESDEFY_SEGMENTER_BACKEND``.
Nothing in the pipeline or API needs to change.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.models.base import BaseSegmenter, SegmentationResult


class MockSegmenter(BaseSegmenter):
    """Returns a fixed elliptical region in the lower-central frame.

    Content-independent by design -- it only provides realistic *geometry* so the
    crop-synthesis and classifier paths run for real. Not a segmentation model.
    """

    name = "mock-segmenter-v0"
    is_mock = True

    def segment(self, image: Image.Image) -> SegmentationResult:
        w, h = image.size
        mask = self._elliptical_region(h, w)
        return SegmentationResult(
            mask=mask,
            coverage=float(mask.mean()),
            backend=self.name,
            is_mock=True,
        )

    @staticmethod
    def _elliptical_region(h: int, w: int) -> np.ndarray:
        """A filled ellipse in the lower-central third (a strip-like eyelid shape)."""
        yy, xx = np.ogrid[:h, :w]
        cy, cx = int(h * 0.62), int(w * 0.50)   # lower-center
        ry, rx = int(h * 0.12), int(w * 0.22)   # wide, short strip
        ellipse = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0
        return ellipse.astype(np.uint8)
