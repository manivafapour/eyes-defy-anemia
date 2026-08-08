"""
Feature-matching + homography alignment for the SECOND tissue type,
forniceal_palpebral (the combined fornix+palpebral view) -- a sibling to
build_aligned_dataset.py (which targets the palpebral-only crop). This is
new data engineering, not a config change: no aligned segmentation dataset
for forniceal_palpebral existed before this script. The alignment
*algorithm* itself is identical and reused as-is (find_homography_alignment,
_alpha_is_functional, _white_background_mask are copied verbatim from
build_aligned_dataset.py, since none of that logic is palpebral-specific)
-- only (a) which crop file is located per patient and (b) the output
location differ.

Locating the forniceal_palpebral crop (find_forniceal_crop_file below) ports
classification/datapreparepipeline/prepare_dataset.py's find_crop_files()
logic -- adapted to read zip entries directly (via zf.namelist()) instead of
an already-extracted folder, matching how phase0_prepare_dataset.py and
build_aligned_dataset.py already read this project's archive.zip. Six Italy
folders (1, 35, 54, 58, 75, 109) have no forniceal conjunctiva photographed
at all -- documented in classification's own report, re-derived here
independently rather than hardcoded, and logged as status="no_forniceal_crop"
(distinct from an attempted-but-failed alignment) rather than silently
skipped.

IMPORTANT, same as build_aligned_dataset.py: this script's output requires a
human visual check before being used for training. The AREA_RATIO_BOUNDS
constant below was tuned empirically for the *palpebral* crop's scale
relative to the raw photo (~3.7x linear / ~14x area for the India zoomed-in
case) -- the forniceal_palpebral crop is a wider, less-zoomed combined view,
so its own scale ratio to the raw photo has NOT been independently verified
and may sit in a different part of the (0.5, 20.0) range, or outside it.
Run this script, then inspect alignment_log.csv's ratio of warped area to
crop area for a handful of real patients (see the module docstring note in
CLAUDE.md Sec 1.4.2 for how this was originally derived) before trusting a
full-batch "ok" result the way it was trusted for palpebral.
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
    ZIP_ROOT,
    find_source_files,
    pad_to_square,
    sanitize_png_bytes,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_splits.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "aligned_raw_forniceal"
IMAGES_OUT_DIR = OUTPUT_ROOT / "images"
MASKS_OUT_DIR = OUTPUT_ROOT / "masks"
LOG_CSV = OUTPUT_ROOT / "alignment_log.csv"

TARGET_SIZE = 256
ALPHA_THRESHOLD = 127

# Same convention-detection thresholds as build_aligned_dataset.py (CLAUDE.md
# Sec 1.4.4) -- unrelated to which crop is being read, so reused unchanged.
OPAQUE_ALPHA_FRACTION_THRESHOLD = 0.99
WHITE_BACKGROUND_THRESHOLD = 245

LOWE_RATIO = 0.75
MIN_GOOD_MATCHES = 4
RANSAC_REPROJ_THRESHOLD = 5.0
MIN_INLIERS_TRUSTED = 15

# NOT YET RE-VERIFIED for this crop -- see module docstring. Carried over
# from build_aligned_dataset.py's palpebral-derived bounds as a starting
# point only.
AREA_RATIO_BOUNDS = (0.5, 20.0)


class AlignmentFailure(Exception):
    pass


# --------------------------------------------------------------------------
# Locate the forniceal_palpebral crop (new -- not needed for palpebral)
# --------------------------------------------------------------------------
def find_forniceal_crop_file(zf: zipfile.ZipFile, country: str, number: int) -> str | None:
    """Returns the zip path of the forniceal_palpebral (combined fornix+
    palpebral) crop for this patient, or None if this patient's forniceal
    conjunctiva was never photographed. Ported from classification's
    find_crop_files(): among PNGs whose filename contains "forniceal"
    (case-insensitive -- covers "palplebral"/"papebral"-style typos same as
    find_source_files does for the plain palpebral crop), the combined view
    is always the longer of the two filenames (the other is the forniceal-
    only, unused third crop type). An unexpected count (not 0 or 2) raises
    rather than guessing, same discipline as find_source_files."""
    prefix = f"{ZIP_ROOT}/{country}/{number}/"
    files = [n for n in zf.namelist() if n.startswith(prefix) and n != prefix]
    files = [f for f in files if "(1)" not in f]

    forniceal_related = [f for f in files if f.lower().endswith(".png") and "forniceal" in f.lower()]

    if len(forniceal_related) == 0:
        return None
    if len(forniceal_related) == 2:
        return max(forniceal_related, key=len)
    raise ValueError(
        f"{country}/{number}: expected 0 or 2 forniceal-related pngs, found {forniceal_related}"
    )


# --------------------------------------------------------------------------
# Feature matching + homography -- verbatim copy from build_aligned_dataset.py
# --------------------------------------------------------------------------
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _enhance(gray: np.ndarray) -> np.ndarray:
    return _CLAHE.apply(gray)


def _match_and_estimate(detector, matcher_norm, raw_gray, crop_gray, crop_alpha_mask):
    kp_crop, des_crop = detector.detectAndCompute(crop_gray, crop_alpha_mask)
    kp_raw, des_raw = detector.detectAndCompute(raw_gray, None)

    if des_crop is None or des_raw is None or len(kp_crop) < 2 or len(kp_raw) < 2:
        return None

    matcher = cv2.BFMatcher(matcher_norm)
    knn_matches = matcher.knnMatch(des_crop, des_raw, k=2)

    good = []
    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        return None

    src_pts = np.float32([kp_crop[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_raw[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    if H is None:
        return None

    n_inliers = int(inlier_mask.sum())
    return {
        "H": H,
        "n_keypoints_crop": len(kp_crop),
        "n_keypoints_raw": len(kp_raw),
        "n_good_matches": len(good),
        "n_inliers": n_inliers,
    }


def find_homography_alignment(raw_gray: np.ndarray, crop_gray: np.ndarray, crop_alpha_mask: np.ndarray) -> dict:
    raw_eq = _enhance(raw_gray)
    crop_eq = _enhance(crop_gray)

    candidates = []
    sift_result = _match_and_estimate(
        cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=20),
        cv2.NORM_L2,
        raw_eq,
        crop_eq,
        crop_alpha_mask,
    )
    if sift_result is not None:
        sift_result["method"] = "SIFT"
        candidates.append(sift_result)

    orb_result = _match_and_estimate(
        cv2.ORB_create(nfeatures=5000, scaleFactor=1.1, nlevels=12),
        cv2.NORM_HAMMING,
        raw_eq,
        crop_eq,
        crop_alpha_mask,
    )
    if orb_result is not None:
        orb_result["method"] = "ORB"
        candidates.append(orb_result)

    if not candidates:
        raise AlignmentFailure("neither SIFT nor ORB found enough good matches")

    result = max(candidates, key=lambda r: r["n_inliers"])

    H = result["H"]
    crop_h, crop_w = crop_gray.shape
    raw_h, raw_w = raw_gray.shape

    corners = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    margin = 0.02 * max(raw_w, raw_h)
    if (
        warped_corners[:, 0].min() < -margin
        or warped_corners[:, 0].max() > raw_w + margin
        or warped_corners[:, 1].min() < -margin
        or warped_corners[:, 1].max() > raw_h + margin
    ):
        raise AlignmentFailure("warped crop corners fall outside the raw photo bounds")

    warped_area = cv2.contourArea(warped_corners.astype(np.float32))
    original_area = crop_w * crop_h
    area_ratio = warped_area / max(original_area, 1)
    if original_area <= 0 or not (AREA_RATIO_BOUNDS[0] <= area_ratio <= AREA_RATIO_BOUNDS[1]):
        raise AlignmentFailure(f"warped area ratio {area_ratio:.3f} outside sanity bounds")

    result["area_ratio"] = area_ratio
    return result


# --------------------------------------------------------------------------
# Mask extraction -- verbatim copy from build_aligned_dataset.py
# --------------------------------------------------------------------------
def _alpha_is_functional(crop_pil: Image.Image) -> bool:
    if crop_pil.mode != "RGBA":
        return False
    alpha = np.array(crop_pil)[..., 3]
    opaque_fraction = (alpha > ALPHA_THRESHOLD).mean()
    return opaque_fraction < OPAQUE_ALPHA_FRACTION_THRESHOLD


def _white_background_mask(crop_rgb: np.ndarray) -> np.ndarray:
    is_background = (crop_rgb >= WHITE_BACKGROUND_THRESHOLD).all(axis=-1)
    return (~is_background).astype(np.uint8) * 255


# --------------------------------------------------------------------------
# Per-patient processing
# --------------------------------------------------------------------------
def process_patient(zf: zipfile.ZipFile, country: str, number: int, patient_id: str) -> dict:
    jpg_name, _palpebral_png_name = find_source_files(zf, country, number)
    forniceal_png_name = find_forniceal_crop_file(zf, country, number)

    if forniceal_png_name is None:
        tqdm.write(f"[skip] {patient_id}: no forniceal_palpebral crop for this patient")
        return {"patient_id": patient_id, "status": "no_forniceal_crop", "n_inliers": None, "mask_source": None}

    raw_img = Image.open(io.BytesIO(zf.read(jpg_name)))
    raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")
    raw_array = np.array(raw_img)

    crop_pil = Image.open(io.BytesIO(sanitize_png_bytes(zf.read(forniceal_png_name))))
    mask_source = "alpha" if _alpha_is_functional(crop_pil) else "white_bg"

    crop_rgba = np.array(crop_pil.convert("RGBA"))
    crop_rgb = crop_rgba[..., :3]

    if mask_source == "alpha":
        crop_alpha = crop_rgba[..., 3]
        crop_alpha_mask = (crop_alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255
        warp_source = crop_alpha
    else:
        crop_alpha_mask = _white_background_mask(crop_rgb)
        warp_source = crop_alpha_mask

    raw_h, raw_w = raw_array.shape[:2]
    crop_h, crop_w = crop_rgb.shape[:2]

    raw_gray = cv2.cvtColor(raw_array, cv2.COLOR_RGB2GRAY)
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    try:
        alignment = find_homography_alignment(raw_gray, crop_gray, crop_alpha_mask)
    except AlignmentFailure as exc:
        tqdm.write(f"[FAIL] {patient_id}: {exc}")
        return {"patient_id": patient_id, "status": f"failed: {exc}", "n_inliers": None, "mask_source": mask_source}

    if alignment["n_inliers"] < MIN_INLIERS_TRUSTED:
        tqdm.write(
            f"[warn] {patient_id}: only {alignment['n_inliers']} RANSAC inliers "
            f"({alignment['method']}) -- low confidence"
        )

    full_mask = cv2.warpPerspective(
        warp_source, alignment["H"], (raw_w, raw_h), flags=cv2.INTER_LINEAR, borderValue=0
    )

    raw_square = pad_to_square(Image.fromarray(raw_array), fill=(0, 0, 0))
    mask_square = pad_to_square(Image.fromarray(full_mask), fill=0)

    raw_final = raw_square.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    mask_final = mask_square.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    raw_final.save(IMAGES_OUT_DIR / f"{patient_id}.jpg", quality=95)
    mask_final.save(MASKS_OUT_DIR / f"{patient_id}.png")

    return {
        "patient_id": patient_id,
        "status": "ok",
        "method": alignment["method"],
        "mask_source": mask_source,
        "n_keypoints_crop": alignment["n_keypoints_crop"],
        "n_keypoints_raw": alignment["n_keypoints_raw"],
        "n_good_matches": alignment["n_good_matches"],
        "n_inliers": alignment["n_inliers"],
        "area_ratio": alignment["area_ratio"],
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
        for _, row in tqdm(splits_df.iterrows(), total=len(splits_df), desc="Aligning forniceal_palpebral (SIFT/ORB + homography)"):
            log_rows.append(
                process_patient(zf, row["country"], int(row["number"]), row["patient_id"])
            )

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_CSV, index=False)

    ok_ids = set(log_df.loc[log_df["status"] == "ok", "patient_id"])
    for out_dir, suffix in [(IMAGES_OUT_DIR, ".jpg"), (MASKS_OUT_DIR, ".png")]:
        for existing_file in out_dir.glob(f"*{suffix}"):
            if existing_file.stem not in ok_ids:
                existing_file.unlink()
                print(f"Removed orphaned output from a previous run: {existing_file}")

    n_ok = int((log_df["status"] == "ok").sum())
    n_no_crop = int((log_df["status"] == "no_forniceal_crop").sum())
    n_failed = len(log_df) - n_ok - n_no_crop
    print(f"\nAligned {n_ok}/{len(log_df)} patients ({n_no_crop} have no forniceal crop, {n_failed} attempted-and-failed).")
    if n_ok:
        ok_rows = log_df[log_df["status"] == "ok"]
        print(ok_rows["method"].value_counts())
        print(
            f"Inlier stats: min={ok_rows['n_inliers'].min()} "
            f"mean={ok_rows['n_inliers'].mean():.1f} max={ok_rows['n_inliers'].max()}"
        )
        print(
            f"Area ratio stats (crop area x this = warped area in raw photo -- "
            f"compare against build_aligned_dataset.py's palpebral numbers, "
            f"NOT assumed to match): min={ok_rows['area_ratio'].min():.3f} "
            f"mean={ok_rows['area_ratio'].mean():.3f} max={ok_rows['area_ratio'].max():.3f}"
        )
        n_low_confidence = int((ok_rows["n_inliers"] < MIN_INLIERS_TRUSTED).sum())
        print(f"Low-confidence alignments (< {MIN_INLIERS_TRUSTED} inliers): {n_low_confidence}")
        print("\nMask source (alpha cutout vs. white-background fallback):")
        print(log_df["mask_source"].value_counts(dropna=False))
    if n_failed:
        print(log_df[(log_df["status"] != "ok") & (log_df["status"] != "no_forniceal_crop")][["patient_id", "status"]])
    print(f"Log written to {LOG_CSV}")
    print("\nNOT YET VERIFIED -- visually inspect a sample (including area_ratio outliers) before trusting for training.")


if __name__ == "__main__":
    main()
