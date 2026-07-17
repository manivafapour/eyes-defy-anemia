"""
Strategic pivot after Phase 3: derive a segmentation mask in the RAW
clinical photo's own coordinate frame, instead of using the palpebral
crop's own (already ~90%+ black) coordinate frame.

Phase 3 showed empirically that a U-Net trained on ConjunctivaSegmentation
Dataset (image = crop's own RGB, mask = crop's own alpha) does not
generalize to raw photos: it almost certainly learned "foreground =
non-black pixel" as a shortcut, since that cue was consistently predictive
on its training distribution but doesn't exist in a normal, fully-lit raw
photo. Retraining on the raw-photo domain requires a real mask aligned to
the raw photo, which Phase 0 never produced (the raw photo and the crop
were padded/resized independently and don't share a pixel grid).

This script recovers that alignment via OpenCV template matching, done on
the ORIGINAL, un-padded, native-resolution images read directly from
archive.zip -- template matching is not scale-invariant, so it has to run
before Phase 0's independent pad-to-square + resize steps break the two
images' relative scale.

For every retained patient (per data/processed/dataset_splits.csv):
1. Read the original raw photo and palpebral crop from archive.zip
   (reusing phase0_prepare_dataset's own file-finding/corruption-repair
   logic, so this stays consistent with Phase 0's exclusion/repair rules).
2. Locate the crop inside the raw photo via cv2.matchTemplate, using the
   crop's own alpha channel as a match mask -- this is necessary because
   the crop's RGB is zeroed everywhere outside the true tissue region
   (documented in CLAUDE.md Sec 1.2), and including those zeroed pixels
   unmasked in the correlation would corrupt the match.
3. Paste the crop's real alpha channel into a blank black canvas the same
   size as the raw photo, at the matched (x, y) offset -- this is the
   mask, now in the raw photo's own coordinate frame.
4. Apply IDENTICAL pad-to-square + Lanczos-resize-to-256 to both the raw
   photo and this new full-scale mask (same input size -> same padding
   decision -> the two stay pixel-aligned).
5. Save to data/processed/aligned_raw/{images,masks}/.
"""

import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from phase0_prepare_dataset import (  # noqa: E402
    ZIP_PATH,
    find_source_files,
    pad_to_square,
    sanitize_png_bytes,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_splits.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "aligned_raw"
IMAGES_OUT_DIR = OUTPUT_ROOT / "images"
MASKS_OUT_DIR = OUTPUT_ROOT / "masks"
LOG_CSV = OUTPUT_ROOT / "alignment_log.csv"

TARGET_SIZE = 256
ALPHA_THRESHOLD = 127
MATCH_METHOD = cv2.TM_CCORR_NORMED  # one of the two methods cv2 supports a mask for
LOW_CONFIDENCE_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# Template matching
# --------------------------------------------------------------------------
def find_crop_location(raw_gray: np.ndarray, crop_gray: np.ndarray, crop_alpha_mask: np.ndarray):
    """Locates crop_gray inside raw_gray via masked normalized cross-
    correlation -- only pixels where crop_alpha_mask is nonzero contribute
    to the match score. Returns ((x, y) top-left corner in raw_gray, and
    the confidence score, higher is better for TM_CCORR_NORMED)."""
    result = cv2.matchTemplate(raw_gray, crop_gray, MATCH_METHOD, mask=crop_alpha_mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val


# --------------------------------------------------------------------------
# Per-patient processing
# --------------------------------------------------------------------------
def process_patient(zf: zipfile.ZipFile, country: str, number: int, patient_id: str) -> dict:
    jpg_name, png_name = find_source_files(zf, country, number)

    raw_img = Image.open(io.BytesIO(zf.read(jpg_name)))
    raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")
    raw_array = np.array(raw_img)

    crop_rgba = np.array(
        Image.open(io.BytesIO(sanitize_png_bytes(zf.read(png_name)))).convert("RGBA")
    )
    crop_rgb = crop_rgba[..., :3]
    crop_alpha = crop_rgba[..., 3]
    crop_alpha_mask = (crop_alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255

    raw_h, raw_w = raw_array.shape[:2]
    crop_h, crop_w = crop_rgb.shape[:2]

    if crop_h > raw_h or crop_w > raw_w:
        return {"patient_id": patient_id, "status": "skipped_crop_larger_than_raw", "confidence": None}

    raw_gray = cv2.cvtColor(raw_array, cv2.COLOR_RGB2GRAY)
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    (x, y), confidence = find_crop_location(raw_gray, crop_gray, crop_alpha_mask)

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        tqdm.write(f"[warn] {patient_id}: low template-match confidence ({confidence:.3f})")

    # The mask, now in the raw photo's own coordinate frame.
    full_mask = np.zeros((raw_h, raw_w), dtype=np.uint8)
    full_mask[y : y + crop_h, x : x + crop_w] = crop_alpha

    # Identical geometric preprocessing for both -- same input size means
    # pad_to_square makes the same padding decision for both, keeping them
    # pixel-aligned through to the final 256x256 output.
    raw_square = pad_to_square(Image.fromarray(raw_array), fill=(0, 0, 0))
    mask_square = pad_to_square(Image.fromarray(full_mask), fill=0)

    raw_final = raw_square.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    mask_final = mask_square.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    raw_final.save(IMAGES_OUT_DIR / f"{patient_id}.jpg", quality=95)
    mask_final.save(MASKS_OUT_DIR / f"{patient_id}.png")

    return {
        "patient_id": patient_id,
        "status": "ok",
        "confidence": confidence,
        "match_x": x,
        "match_y": y,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "raw_w": raw_w,
        "raw_h": raw_h,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits_df = pd.read_csv(SPLITS_CSV)
    log_rows = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for _, row in tqdm(splits_df.iterrows(), total=len(splits_df), desc="Aligning raw photos + masks"):
            log_rows.append(
                process_patient(zf, row["country"], int(row["number"]), row["patient_id"])
            )

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_CSV, index=False)

    n_ok = int((log_df["status"] == "ok").sum())
    n_skipped = len(log_df) - n_ok
    print(f"\nAligned {n_ok}/{len(log_df)} patients ({n_skipped} skipped).")
    if n_ok:
        ok_confidence = log_df.loc[log_df["status"] == "ok", "confidence"]
        print(
            f"Confidence stats: min={ok_confidence.min():.4f} "
            f"mean={ok_confidence.mean():.4f} max={ok_confidence.max():.4f}"
        )
        n_low_confidence = int((ok_confidence < LOW_CONFIDENCE_THRESHOLD).sum())
        print(f"Low-confidence matches (< {LOW_CONFIDENCE_THRESHOLD}): {n_low_confidence}")
    print(f"Log written to {LOG_CSV}")


if __name__ == "__main__":
    main()
