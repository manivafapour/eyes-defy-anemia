"""
Feature-matching + homography alignment (v2) -- replaces the earlier
matchTemplate-based approach, which was proven wrong: a manual visual check
of India_071 showed the "aligned" mask floating on the sclera (white of the
eye) instead of the lower eyelid, despite a high (0.995) normalized
cross-correlation confidence score. matchTemplate is a pure appearance
correlation with no geometric constraint -- it can lock onto a *different*
region that happens to correlate well (skin tone, vessel texture, general
color/lighting), and a high score does not guarantee a *correct* location.
The automated "is the mask non-blank" sanity check from the previous
attempt could not catch this failure mode at all, since the mask was very
much non-blank -- just in the wrong place.

Feature matching + RANSAC homography is structurally different: it requires
many keypoint correspondences between the crop and the raw photo to agree
on a *single consistent geometric transform* (translation + rotation +
scale + perspective). A coincidental appearance match at the wrong location
essentially never produces enough mutually-consistent keypoint
correspondences to pass RANSAC, which is exactly the robustness property
matchTemplate lacked.

For every retained patient (per data/processed/dataset_splits.csv):
1. Read the original raw photo and palpebral crop from archive.zip
   (reusing phase0_prepare_dataset's own file-finding/corruption-repair
   logic, so this stays consistent with Phase 0's exclusion/repair rules).
2. Determine which convention this patient's crop uses to delimit tissue,
   and build the foreground mask accordingly (CLAUDE.md Sec 1.4.4):
     - Normal case: the alpha channel is a real cutout (RGB zeroed outside
       it) -- mask = alpha > ALPHA_THRESHOLD.
     - Fallback case: the alpha channel is uniformly (or near-uniformly)
       opaque, or absent entirely -- this means the source file instead
       uses an OPAQUE WHITE background to delimit tissue, so alpha carries
       no shape information at all. Mask = NOT-near-white RGB instead
       (_white_background_mask). Detected automatically per-patient via
       _alpha_is_functional(), not a hardcoded patient list.
3. Detect SIFT keypoints/descriptors in the crop (restricted to that mask
   -- background pixels have no real texture there to (mis)match against)
   and in the full raw photo.
4. Match descriptors (BFMatcher + Lowe's ratio test), then estimate a
   homography via cv2.findHomography(..., cv2.RANSAC) from the surviving
   correspondences. Falls back to ORB if SIFT does not find enough matches.
5. Warp the crop's mask (real alpha channel, or the white-background
   fallback mask) through that homography into the raw photo's coordinate
   frame via cv2.warpPerspective -- this is the mask, now correctly
   scaled/rotated/positioned in the raw photo's own grid.
6. Apply IDENTICAL pad-to-square + Lanczos-resize-to-256 to both the raw
   photo and this new full-scale mask (same input size -> same padding
   decision -> the two stay pixel-aligned).
7. Save to data/processed/aligned_raw/{images,masks}/, with alignment_log.csv
   additionally recording which mask_source each patient used.

IMPORTANT: this script's output requires a human visual check
(notebooks/verify_alignment_sanity_check.ipynb) before being used for
training -- see that notebook's own verification, including India_071
specifically (the patient that exposed the earlier matchTemplate
approach's failure). The white-background fallback (step 2 above) was
itself added after a *separate* human visual check
(notebooks/find_corrupted_masks.ipynb) caught 30 patients whose masks
covered ~75% of the frame under the old alpha-only logic -- any full
regeneration after this fix should be re-verified the same way before
being trusted for training.
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

# A subset of patients' source palpebral crop PNGs (all confirmed Italy so
# far) use an OPAQUE WHITE background instead of alpha transparency to
# delimit the tissue -- i.e. a different source-data convention, not missing
# data. Their alpha channel is uniformly (or near-uniformly) opaque and
# therefore carries no shape information at all, even though it never
# raises/errors anywhere -- (alpha > ALPHA_THRESHOLD) silently evaluates to
# "the entire rectangle is foreground". Verified across the full 217-patient
# source data: legitimate alpha-cutout patients top out at ~13% opaque
# fraction (Italy_113); every affected patient sits at exactly 100%. Any
# threshold between those two clusters is safe -- 0.99 is used here for a
# wide margin. notebooks/find_corrupted_masks.ipynb has the full
# investigation (root cause, before/after visual + quantitative
# verification on Italy_022/Italy_004).
OPAQUE_ALPHA_FRACTION_THRESHOLD = 0.99
WHITE_BACKGROUND_THRESHOLD = 245  # RGB >= this in all 3 channels counts as background

LOWE_RATIO = 0.75
MIN_GOOD_MATCHES = 4          # cv2.findHomography's hard minimum
RANSAC_REPROJ_THRESHOLD = 5.0  # pixels, at native (un-padded) resolution
MIN_INLIERS_TRUSTED = 15       # below this, flag as low-confidence but still write output

# NOTE: this was initially set to (0.5, 2.0) on the assumption that crop and
# raw photo must be near-unit-scale (same camera capture). That assumption
# was WRONG -- direct visual inspection of India_071 and India_001 (the
# patient that exposed the original matchTemplate failure) confirmed the
# mask lands correctly on the lower eyelid at a real, consistent ~3.7x
# linear scale (~14x area), most likely because the palpebral crop was
# captured as a separate, more zoomed-in shot rather than a simple 1:1
# sub-crop of the raw photo. 73/76 of India_071's keypoint correspondences
# independently agreed on this same scale ratio (SIFT keypoint .size
# comparison, not just the final homography) -- too consistent to be a
# coincidental false match. Bounds widened to comfortably cover both the
# ~1.0x case (Italy_001, whose "crop" is the same size as its raw photo)
# and the ~14x case (India), while still catching truly degenerate results.
AREA_RATIO_BOUNDS = (0.5, 20.0)


class AlignmentFailure(Exception):
    """Raised (and caught) whenever a patient cannot be aligned at all."""


# --------------------------------------------------------------------------
# Feature matching + homography
# --------------------------------------------------------------------------
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _enhance(gray: np.ndarray) -> np.ndarray:
    """Conjunctiva tissue is naturally low-contrast (subtle blood-vessel
    texture on a fairly uniform pink/red background) -- plain grayscale
    starves SIFT/ORB of detectable keypoints (as few as ~30-70 in a
    30k+ pixel tissue region). CLAHE brings local contrast up enough to
    multiply keypoint yield several-fold (verified: 34 -> 427 keypoints
    on India_071's crop with CLAHE + a looser SIFT contrast threshold)."""
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
            continue  # too few raw keypoints to find a 2nd nearest neighbor
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
    """Runs both SIFT and ORB (each on CLAHE-contrast-enhanced grayscale)
    and keeps whichever produces more RANSAC inliers -- not just whichever
    clears the bare minimum first, since a low-keypoint-count match can look
    superficially valid (passes findHomography) while actually being an
    unstable, near-degenerate fit. Raises AlignmentFailure if neither
    detector works, or if the resulting homography fails a geometric
    sanity check (warped crop corners must land within the raw photo and
    span a plausible, near-unit-scale area -- see AREA_RATIO_BOUNDS)."""
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

    margin = 0.02 * max(raw_w, raw_h)  # small tolerance for sub-pixel/rounding overshoot
    if (
        warped_corners[:, 0].min() < -margin
        or warped_corners[:, 0].max() > raw_w + margin
        or warped_corners[:, 1].min() < -margin
        or warped_corners[:, 1].max() > raw_h + margin
    ):
        raise AlignmentFailure("warped crop corners fall outside the raw photo bounds")

    warped_area = cv2.contourArea(warped_corners.astype(np.float32))
    original_area = crop_w * crop_h
    if original_area <= 0 or not (AREA_RATIO_BOUNDS[0] <= warped_area / original_area <= AREA_RATIO_BOUNDS[1]):
        raise AlignmentFailure(
            f"warped area ratio {warped_area / max(original_area, 1):.3f} outside sanity bounds"
        )

    return result


# --------------------------------------------------------------------------
# Mask extraction: alpha-cutout convention, or white-background fallback
# --------------------------------------------------------------------------
def _alpha_is_functional(crop_pil: Image.Image) -> bool:
    """True if this crop's alpha channel actually encodes a tissue cutout
    (real, non-trivial transparency) rather than being uniformly opaque or
    absent entirely. A crop with no alpha channel at all (plain RGB) is
    treated the same as a uniformly-opaque one -- both carry zero shape
    information, just via different underlying causes."""
    if crop_pil.mode != "RGBA":
        return False
    alpha = np.array(crop_pil)[..., 3]
    opaque_fraction = (alpha > ALPHA_THRESHOLD).mean()
    return opaque_fraction < OPAQUE_ALPHA_FRACTION_THRESHOLD


def _white_background_mask(crop_rgb: np.ndarray) -> np.ndarray:
    """Foreground = NOT near-white. Fallback for patients whose source crop
    delimits tissue with an opaque white background instead of alpha
    transparency -- see OPAQUE_ALPHA_FRACTION_THRESHOLD above."""
    is_background = (crop_rgb >= WHITE_BACKGROUND_THRESHOLD).all(axis=-1)
    return (~is_background).astype(np.uint8) * 255


# --------------------------------------------------------------------------
# Per-patient processing
# --------------------------------------------------------------------------
def process_patient(zf: zipfile.ZipFile, country: str, number: int, patient_id: str) -> dict:
    jpg_name, png_name = find_source_files(zf, country, number)

    raw_img = Image.open(io.BytesIO(zf.read(jpg_name)))
    raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")
    raw_array = np.array(raw_img)

    crop_pil = Image.open(io.BytesIO(sanitize_png_bytes(zf.read(png_name))))
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
        "method": alignment["method"],
        "mask_source": mask_source,
        "n_keypoints_crop": alignment["n_keypoints_crop"],
        "n_keypoints_raw": alignment["n_keypoints_raw"],
        "n_good_matches": alignment["n_good_matches"],
        "n_inliers": alignment["n_inliers"],
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
        for _, row in tqdm(splits_df.iterrows(), total=len(splits_df), desc="Aligning (SIFT/ORB + homography)"):
            log_rows.append(
                process_patient(zf, row["country"], int(row["number"]), row["patient_id"])
            )

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_CSV, index=False)

    # process_patient() only WRITES output files for patients that succeed --
    # a patient that succeeded in a previous run but fails in this one (e.g.
    # after a logic change like CLAUDE.md Sec 1.4.4's mask-source fix) would
    # otherwise leave its stale, no-longer-valid output files on disk
    # indefinitely, since nothing ever deletes them. Caught for real: a
    # re-run after the 1.4.4 fix left Italy_026's old (wrong, alpha-derived)
    # image/mask on disk even though it now correctly fails alignment.
    ok_ids = set(log_df.loc[log_df["status"] == "ok", "patient_id"])
    for out_dir, suffix in [(IMAGES_OUT_DIR, ".jpg"), (MASKS_OUT_DIR, ".png")]:
        for existing_file in out_dir.glob(f"*{suffix}"):
            if existing_file.stem not in ok_ids:
                existing_file.unlink()
                print(f"Removed orphaned output from a previous run: {existing_file}")

    n_ok = int((log_df["status"] == "ok").sum())
    n_failed = len(log_df) - n_ok
    print(f"\nAligned {n_ok}/{len(log_df)} patients ({n_failed} failed).")
    if n_ok:
        ok_rows = log_df[log_df["status"] == "ok"]
        print(ok_rows["method"].value_counts())
        print(
            f"Inlier stats: min={ok_rows['n_inliers'].min()} "
            f"mean={ok_rows['n_inliers'].mean():.1f} max={ok_rows['n_inliers'].max()}"
        )
        n_low_confidence = int((ok_rows["n_inliers"] < MIN_INLIERS_TRUSTED).sum())
        print(f"Low-confidence alignments (< {MIN_INLIERS_TRUSTED} inliers): {n_low_confidence}")
        print("\nMask source (alpha cutout vs. white-background fallback, CLAUDE.md Sec 1.4.4):")
        print(log_df["mask_source"].value_counts(dropna=False))
    if n_failed:
        print(log_df[log_df["status"] != "ok"][["patient_id", "status"]])
    print(f"Log written to {LOG_CSV}")
    print("\nNOT YET VERIFIED -- run notebooks/verify_alignment_sanity_check.ipynb and visually confirm before training.")


if __name__ == "__main__":
    main()
