"""Shared image preprocessing -- the single source of truth.

*** PARITY-CRITICAL MODULE ***

The classifier (Stage 2) was trained on conjunctiva crops that had been flattened
onto a solid black background, padded to a square, and resized. In production the
crop is *synthesized* from the raw photo + Stage-1 mask instead of coming from a
human-made source crop. If the synthesis here drifts from the training-time
preprocessing, the model sees an out-of-distribution input and accuracy silently
degrades (train/serve skew).

This is the *third* place in this repo where crop preprocessing lives (Segmentation
and classification each hit the same white-background bug independently). Keep the
logic here only, and lock it with a golden parity test in Phase 0 -- compare these
synthesized crops against the 217 archive source crops -- before trusting any
production number.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

from app.core.exceptions import InvalidImageError


def load_image(data: bytes) -> Image.Image:
    """Decode upload bytes into an EXIF-corrected RGB image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a client error
        raise InvalidImageError("Uploaded file is not a readable image.") from exc
    image = ImageOps.exif_transpose(image)  # honor sensor orientation
    return image.convert("RGB")


def pad_to_square(image: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Center the image on a square canvas (black fill), without distortion."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def synthesize_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Turn (raw image, tissue mask) into the classifier's expected crop.

    Mirrors the training-time crop convention:

      1. zero out every non-tissue pixel   (background -> black)
      2. crop to the mask's bounding box
      3. pad to a square black canvas

    The final resize + ImageNet normalization are owned by the classifier backend
    (they belong to the model, mirroring how the training ``Dataset`` applied them),
    not to this function.

    An empty mask yields an all-black image; the caller's coverage gate should have
    already rejected that case.
    """
    rgb = np.asarray(image, dtype=np.uint8)
    m = mask.astype(bool)
    if m.shape != rgb.shape[:2]:
        raise ValueError(f"Mask shape {m.shape} does not match image {rgb.shape[:2]}.")

    masked = np.zeros_like(rgb)
    masked[m] = rgb[m]

    ys, xs = np.where(m)
    if ys.size == 0:
        return Image.fromarray(masked)  # empty mask -> all black

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = Image.fromarray(masked[y0:y1, x0:x1])
    return pad_to_square(cropped)
