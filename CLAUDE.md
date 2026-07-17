# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It also serves as the primary methodology reference for the project's written thesis — entries below are written for scientific accuracy and should be treated as authoritative over any prior/looser description of the same step.

**Session-start instruction:** before doing anything else in a new session, silently read `.project_memory/01_roadmap.md`, `.project_memory/02_current_status.md`, and `.project_memory/03_tech_stack_and_rules.md` to load current project context. After completing a major task or making an architectural decision, update the relevant file(s) in `.project_memory/` before moving on. `.project_memory/` is the lightweight, frequently-updated working-memory system; this file (`CLAUDE.md`) remains the authoritative, thesis-grade methodology record — the two should stay consistent, but `CLAUDE.md` is the one held to scientific-accuracy standards.

## Project Overview

EYES-DEFY-ANEMIA is an eye-image-based anemia detection project using photographs of the palpebral conjunctiva (inner eyelid lining). The pipeline has two downstream modeling tasks sharing one data foundation: (1) segmentation of the palpebral conjunctiva region, and (2) binary anemia classification from the raw eye photo. Source data: 218 patient folders (India + Italy) inside `archive.zip`, each with a hemoglobin (Hgb) reading, gender, age, a raw eye photo, and a palpebral conjunctiva crop.

---

## Phase 0 — Data Engineering (`scripts/phase0_prepare_dataset.py`)

### 0.1 Metadata unification and label derivation
India and Italy metadata (originally separate `.xlsx` sheets inside `archive.zip`) were merged into a single table. Hemoglobin values were parsed to handle Italy's comma-decimal locale format (e.g. `"15,1"` → `15.1`). Anemia status is a derived binary label from WHO diagnostic thresholds for non-pregnant adults:

- Male: anemic if Hgb < 13.0 g/dL
- Female: anemic if Hgb < 12.0 g/dL

**Stated limitation:** the source data has no pregnancy field, so all female patients are assumed non-pregnant. This is an acknowledged simplification, not an oversight — WHO pregnancy-specific thresholds could not be applied.

### 0.2 Patient exclusion criteria
Two exclusion rules were applied prior to any modeling:
1. **Missing/invalid Hgb** — a row with no parseable hemoglobin value cannot be labeled and is dropped.
2. **`ELIMINATO` flag** — the source spreadsheets mark some rows invalid via the literal string `"ELIMINATO"`, which can appear in any column (verified to occur as far as column G for one Italy record), so all columns were scanned, not just the core five.

Under this rule, **`Italy_093`** was dropped (`ELIMINATO` flag present). This is distinct from `India_093`, which is a valid, included record (Hgb 12.7 g/dL, male, anemic under the 13.0 g/dL threshold) — the two are unrelated patients that happen to share a numeric ID across countries.

Final retained cohort: **217 patients** (95 India, 122 Italy).

### 0.3 Image and mask extraction
For each retained patient, two source files were located per folder using a typo-tolerant rule (any single `.jpg` = raw photo; the sole `.png` *not* containing "forniceal" in its filename = the palpebral crop; verified against all 218 folders with zero rule violations). Each of the two images was independently:
1. EXIF-transposed (corrects landscape/portrait sensor orientation mismatches).
2. Padded to a square canvas (black fill, `(0,0,0)` or `(0,0,0,0)` depending on channel mode) around its original aspect ratio, centering the content.
3. Resized to **256×256** via Lanczos resampling.

The raw photo is saved to `data/processed/images/{patient_id}.jpg`; the palpebral crop is saved to `data/processed/masks/{patient_id}_palpebral.png`. **Important:** these two files are padded/resized independently of one another, so they do **not** share a common pixel coordinate grid (see §1.2 for why this matters for segmentation).

### 0.4 Corrupted PNG chunk repair
63 of the 217 palpebral crop PNGs (~29%) had a corrupted CRC on their ancillary `iCCP` chunk (embedded ICC color profile), which causes Pillow to refuse to open the file at all. Per-chunk CRC verification confirmed the corruption was isolated to `iCCP` in every case — the critical chunks (`IHDR`, `PLTE`, `IDAT`, `IEND`, i.e. the actual pixel data) always passed CRC verification. The repair strips only chunks with a failing CRC that are *not* in the critical set; a critical-chunk CRC failure would raise an exception rather than being silently discarded, since that would indicate genuine pixel-data corruption rather than a discardable metadata defect.

### 0.5 Demographic composition and class-balance bias
Anemia prevalence differs sharply by country and is confounded with gender composition. Measured on the full 217-patient cohort:

| Subgroup | n | Anemic rate |
|---|---|---|
| India, female | 46 | **87.0%** |
| India, male | 49 | 57.1% |
| Italy, female | 40 | **32.5%** |
| Italy, male | 82 | 12.2% |
| India, overall | 95 | 71.6% |
| Italy, overall | 122 | 18.9% |

**Note on citing this figure:** "India 87% vs Italy 32%" refers specifically to the *female* subgroup rate, not the overall country rate (India 71.6% vs Italy 18.9% overall). Italy's cohort also skews strongly male (82M/40F) relative to India's near-even split (49M/46F), which is a second, compounding source of demographic imbalance. This country×label (and implicitly country×gender) confound is the direct motivation for the stratification strategy in Phase 1 (§1.1).

---

## Phase 1 — Data Pipeline Construction (`scripts/dataset.py`)

### 1.1 Patient-level stratified splitting: rationale and procedure
A naive random 70/15/15 split risks each split independently drifting in country/label composition by chance — e.g. a validation set that is disproportionately Italy-heavy and non-anemic would understate true model difficulty, since Italy's overall anemia rate (18.9%) is nearly 4× lower than India's (71.6%). To prevent this, splitting is stratified on the compound key `country + "_" + anemic_label`, producing four strata: `India_1`, `India_0`, `Italy_1`, `Italy_0`.

Measured stratum sizes (of 217): `Italy_0` = 99, `India_1` = 68, `India_0` = 27, `Italy_1` = 23. The smallest stratum (23) is comfortably large enough that a 15% slice (≈3–4 patients) remains non-degenerate.

The split is computed as two sequential stratified splits (both stratified on the same compound key, both with `random_state=42` for reproducibility):
1. 70% / 30% → `train` / `temp`
2. `temp` split 50% / 50% → `val` / `test`

Since 30% × 50% = 15%, this yields the target 70/15/15 proportions while stratifying every cut. All splitting is at the **patient level** (each `patient_id` is atomic — no patient's image appears in more than one split), which is the correct unit of independence here since there is exactly one image pair per patient.

Verified realized split (`data/processed/dataset_splits.csv`):

| Split | n | % of total | India (n, rate) | Italy (n, rate) |
|---|---|---|---|---|
| train | 151 | 69.6% | 66, 71.2% | 85, 18.8% |
| val | 33 | 15.2% | 14, 71.4% | 19, 21.1% |
| test | 33 | 15.2% | 15, 73.3% | 18, 16.7% |

Each split's per-country anemia rate closely tracks the population rate (India ≈71–73%, Italy ≈17–21% across all three splits), confirming the stratification achieved its purpose.

### 1.2 `ConjunctivaSegmentationDataset`: mask construction and its limitation
This dataset returns `(image, mask)` pairs, both sourced from the **same** file, `data/processed/masks/{patient_id}_palpebral.png`:
- `image` = the RGB channels of the palpebral crop.
- `mask` = that file's own alpha channel, binarized (`alpha > 127 → 1.0`, else `0.0`), shaped `[1, H, W]`.

**Why not pair the mask with the raw eye photo instead:** the raw photo (`data/processed/images/`) and the palpebral crop (`data/processed/masks/`) were padded and resized *independently* in Phase 0 (§0.3) and therefore do not share a pixel coordinate grid. There is no recoverable spatial correspondence between "pixel (x,y) in the raw photo" and "pixel (x,y) in the crop" without re-deriving crop offsets from the original, un-padded source images — out of scope for Phase 1. The RGB+alpha pairing from the same file is therefore the only pairing in the current pipeline that is pixel-aligned by construction.

**Documented limitation:** inspection of the source files (verified on `India_001`) showed the RGB channels are already zeroed everywhere the alpha channel is 0 (mean RGB ≈ 0.0000165 in the alpha=0 region, vs. a real tissue-colored mean of `[164, 71, 102]` where alpha>0). This means the segmentation input already visually encodes the answer to a large extent (a naive threshold on RGB≈0 nearly reconstructs the mask), making the task easier/leakier than an independent-signal segmentation problem. This was a deliberate, discussed trade-off to get a pixel-aligned, working pipeline now rather than deferring Phase 1 to solve crop re-registration; it should be reported as a limitation of the segmentation benchmark, not presented as an unconfounded test of tissue-segmentation capability.

`AnemiaClassificationDataset` has no such caveat: it returns `(image, label)` from the raw eye photo and the WHO-threshold label directly, with no derived/aligned mask involved.

### 1.3 Synchronized data augmentation (`albumentations` 2.0.8)
Training transforms apply `HorizontalFlip(p=0.5)` and `Rotate(limit=±15°, p=0.5)` identically to `image` and `mask` in a single `albumentations.Compose` call (`transform(image=..., mask=...)`), which is the mechanism that keeps the geometric augmentation spatially synchronized between the two.

The two targets use different interpolation during rotation, by design:
- `image` → linear interpolation (`cv2.INTER_LINEAR`), suitable for continuous-valued RGB pixels.
- `mask` → **nearest-neighbor interpolation** (`cv2.INTER_NEAREST`), which is required to preserve the mask's binary `{0.0, 1.0}` values — linear/bilinear interpolation would introduce fractional intermediate values at the rotated edge, corrupting the "Mask Channel Enforcement" requirement that masks stay strictly binary.

`Normalize` (ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`) is applied only to the `image` target — by default, `albumentations.Normalize` does not register against the `mask` target, so the mask is left untouched at `{0.0, 1.0}` through the full pipeline.

Validation/test transforms are deterministic only (`Resize` → `Normalize` → `ToTensorV2`) — no random flip or rotation, so evaluation metrics are computed on a fixed, non-stochastic view of each sample.

### 1.4 Data-centric fix: raw-photo-aligned segmentation dataset (`scripts/build_aligned_dataset.py`)
**Motivation:** empirical testing of a model trained on §1.2's dataset (image = palpebral crop's own RGB, mask = that file's own alpha) showed it does not generalize to raw eye photos — predictions came out as near-full-frame regardless of input, consistent with the model having learned "foreground = non-black pixel" as a shortcut, since that cue is reliably predictive on §1.2's mostly-black training input but does not exist in a normal, fully-lit raw photo. Training a model that generalizes to raw photos requires a real mask *in the raw photo's own coordinate frame* — which Phase 0 never produced, since the raw photo and the crop were padded/resized independently and don't share a pixel grid (§0.3).

#### 1.4.1 v1 (superseded): template matching — proven wrong, abandoned
The first implementation located the crop inside the raw photo via `cv2.matchTemplate` (`TM_CCORR_NORMED`, masked by the crop's own alpha channel). It reported excellent-looking aggregate statistics — 217/217 "aligned," confidence range 0.952–0.998 — and passed 5 hand-picked visual spot-checks. **It was still wrong.** A later manual review of `India_071` (not one of the 5 originally spot-checked) found the "aligned" mask floating on the sclera (white of the eye) instead of the lower eyelid — a complete, spatially-wrong placement, despite a 0.995 confidence score. The lesson: `matchTemplate` is a pure appearance-correlation score with no geometric constraint, so a high score never guaranteed a correct location — it can lock onto a *different* region that coincidentally correlates well (similar skin tone, vessel texture, lighting). The "is the mask blank" automated sanity check from this version (`notebooks/verify_alignment_sanity_check.ipynb`, first version) could not catch this failure mode at all, since the mask was very much non-blank — just in the wrong place. **This entire approach, and the `aligned_raw` data it produced, was discarded.**

#### 1.4.2 v2 (current): SIFT/ORB feature matching + RANSAC homography
Feature matching + homography is structurally different from template matching: it requires many keypoint correspondences between the crop and the raw photo to agree on a *single consistent geometric transform*, not one global appearance score. A coincidental appearance match at the wrong location essentially never produces enough mutually-consistent correspondences to pass RANSAC.

1. As in v1, the *original* (un-padded, native-resolution) raw photo and palpebral crop are read directly from `archive.zip` (reusing Phase 0's `find_source_files`/`sanitize_png_bytes`), before Phase 0's pad-to-square + resize would break their true relative scale.
2. **Keypoints/descriptors:** SIFT is run on CLAHE-contrast-enhanced grayscale (`cv2.createCLAHE`) for both images — conjunctiva tissue is naturally low-contrast, and plain grayscale starves SIFT of keypoints (verified: 34 → 427 keypoints on `India_071`'s crop after adding CLAHE + a looser SIFT `contrastThreshold=0.01`). Keypoint detection on the crop is restricted to its alpha>0 region (its RGB is zeroed elsewhere, so there is no real texture to search). ORB is also run as a second candidate; whichever detector yields more RANSAC inliers is kept (not just whichever clears the bare minimum first).
3. **Matching:** `cv2.BFMatcher` + Lowe's ratio test (0.75), then `cv2.findHomography(..., cv2.RANSAC, ransacReprojThreshold=5.0)` on the surviving correspondences.
4. **Geometric sanity check (rejects rather than silently accepting a bad fit):** the crop's 4 corners are warped through the estimated homography and must land within the raw photo's bounds, and the warped area must fall within `(0.5, 20.0)×` the original crop area. This bound was tuned *empirically*, not assumed: an early version used `(0.5, 2.0)` on the assumption crop and raw must be near-unit-scale (same camera capture) — that assumption was itself wrong. Direct visual + statistical inspection of `India_071` and `India_001` confirmed the mask lands correctly at a real, consistent ~3.7× linear scale (73/76 of `India_071`'s keypoint correspondences independently agreed on this same scale ratio via SIFT's own `kp.size`, too consistent to be a coincidental false match) — most likely because the palpebral crop was captured as a separate, more zoomed-in shot rather than a simple 1:1 sub-crop of the raw photo. The bound was widened to cover both this ~14× case (India) and the ~1× case (`Italy_001`, whose "crop" is the same size as its raw photo).
5. The crop's real alpha channel is warped through the homography (`cv2.warpPerspective`) directly into the raw photo's coordinate frame — this is the new mask.
6. Identical geometric preprocessing (pad-to-square, Lanczos resize to 256×256) is applied to *both* the raw photo and this new full-scale mask, exactly as in v1.
7. Output: `data/processed/aligned_raw/images/{patient_id}.jpg`, `data/processed/aligned_raw/masks/{patient_id}.png`, plus `alignment_log.csv` (per-patient method, keypoint/match/inlier counts).

**Results (full 217-patient run):** 202/217 aligned, **15 honest failures** (rejected by the geometric sanity checks above, rather than silently producing a wrong result) — 13 "corners outside raw photo bounds," 2 "area ratio outside bounds." All 202 successes used SIFT (ORB fallback was never the better candidate). Inlier counts range 5–2014 (mean 311); 9 flagged low-confidence (<15 inliers). Of the produced masks, 0/202 are blank or near-blank (min positive pixels 1793).

**Visual verification:** `India_071` (the patient that exposed v1's failure) now shows the mask precisely on the lower eyelid, matching the original crop's shape. Additional spot-checks (`India_001`, `India_029`, `Italy_097`) also show correct, anatomically-sensible placement. **This is not yet declared a fully-verified dataset** — per explicit instruction, no training may proceed until the project author has personally reviewed `notebooks/verify_alignment_sanity_check.ipynb`'s full output (all 10 sampled patients, including `India_071`, plus the full-dataset failure list) and confirmed it.

**New dataset class:** `AlignedConjunctivaSegmentationDataset` (`scripts/dataset.py`) reads `(image, mask)` from `data/processed/aligned_raw/{images,masks}/` — image is the *full raw photo*, mask is a genuinely pixel-aligned tissue mask in that same frame. It is additive, not a replacement: `ConjunctivaSegmentationDataset` (§1.2) is unchanged and still valid for its original purpose (and for comparison); `AlignedConjunctivaSegmentationDataset` is the dataset to use going forward for a segmentation model intended to generalize to raw photos. Both are wired into `get_dataloaders()` (`seg_*` vs. `aligned_seg_*` keys) using the same `get_train_transforms()`/`get_eval_transforms()` pipeline (§1.3) — no new transform logic was needed, since both datasets share the same 256×256 RGB image / single-channel binary mask contract.

#### 1.4.3 Decision: drop the 15 failed patients rather than manually annotate them
A `cv2.selectROI`-based manual bounding-box annotation tool was built and tried for the 15 failed patients. **This was rejected** in favor of simply excluding them: 202 correctly-aligned patients were judged sufficient, and forcing an alignment onto extreme edge cases risked introducing noisy training targets (a hand-drawn rectangle is a much cruder ground truth than the SIFT/RANSAC-derived mask the other 202 patients have, and would be inconsistent within the same dataset). The manual-annotation script was deleted.

`AlignedConjunctivaSegmentationDataset` now filters to only patients with `status == "ok"` in `alignment_log.csv`, joined against `dataset_splits.csv` **without modifying that shared CSV** — `ConjunctivaSegmentationDataset` and `AnemiaClassificationDataset` still see all 217 patients, since the 15 excluded here are perfectly valid data for those. Verified: `aligned_seg_train`/`val`/`test` now return 143/28/31 (202 total), while `seg_*`/`cls_*` are unchanged at 151/33/33 (217 total) — and a real batch fetch through `get_dataloaders()["aligned_seg_train"]` still returns correct shapes/mask range.

**Side effect worth noting:** the 15 excluded patients are disproportionately India (13 India, 2 Italy) — not surprising, since India's crops are the smaller, harder-to-match ones (§1.4.2). This shifts the aligned-only subset's country balance slightly further toward Italy than the original 70/15/15 stratification (§1.1) targeted (e.g. train: India 66→59, Italy 85→84). This is a known, accepted trade-off of the exclusion decision, not a bug — worth remembering when interpreting any per-country results from a model trained on this subset.

**Update:** `scripts/trainer_engine.py` now accepts a `dataset_cls` parameter (§2.6, §3.2), and three new sibling entry-point scripts (`train_standard_unet_aligned.py`, `train_attention_unet_aligned.py`, `train_resunet_aligned.py`) target `AlignedConjunctivaSegmentationDataset`. The original three entry-point scripts are untouched and still train against `ConjunctivaSegmentationDataset` by default. **None of the three `_aligned` entry-point scripts have been run yet** — this is infrastructure only; no aligned-dataset training results exist as of this entry. Getting `data/processed/aligned_raw/{images,masks}/` onto Kaggle (they're `.gitignore`'d, so `git pull` alone won't bring them over) and updating the Kaggle-side copy/setup script are separate, not-yet-done steps.

---

## Computational Environment / Hardware Setup

- GPU: **NVIDIA GeForce RTX 4050 Laptop GPU** (driver reports CUDA capability up to 13.1 via `nvidia-smi`).
- An initial unpinned `pip install torch` resolved to a **CPU-only** build (`torch 2.13.0+cpu`) — no GPU acceleration.
- This was superseded by an explicit CUDA-enabled reinstall: **`torch 2.13.0+cu130`**, **`torchvision 0.28.0+cu130`**, **`torchaudio 2.11.0+cu130`** (built against the CUDA 13.0 runtime, which the 13.1-capable driver runs without issue). `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` both confirmed against the RTX 4050 in this environment.
- Python: 3.14.6, project virtual environment at `venv/`.
- Other pinned/verified library versions in this environment: pandas 3.0.3, numpy 2.5.1, Pillow 12.3.0, openpyxl 3.1.5, albumentations 2.0.8, opencv-python (`cv2`) 5.0.0, scikit-learn 1.9.0, optuna 4.9.0.
- `requirements.txt` has since been regenerated via `pip freeze` and now tracks the full installed environment, including `torch==2.13.0+cu130`, `torchvision==0.28.0+cu130`, `torchaudio==2.11.0+cu130`, `albumentations==2.0.8`, and `optuna==4.9.0`.

---

## Phase 2 — Segmentation Model Architecture (`models/segmentation/unet.py`)

### 2.1 Architecture specification
A standard U-Net (Ronneberger et al., 2015 topology) with a symmetrical encoder-decoder and skip connections:

- **Input:** 3×256×256 RGB.
- **Encoder:** 4 downsampling stages, each `MaxPool2d(2)` → `DoubleConv`. Channel progression `64 → 128 → 256 → 512 → 1024` (bottleneck), spatial resolution `256 → 128 → 64 → 32 → 16`.
- **`DoubleConv` block:** `(Conv2d(3×3, padding=1, bias=False) → BatchNorm2d → ReLU) × 2`.
- **Padding choice:** *same*-padding (`padding=1`) rather than the original paper's valid-padding. Because 256 is exactly divisible by 2⁴ = 16, every skip connection matches its decoder-stage spatial size exactly with no cropping required — a simplification over the original paper's design, made possible by the fixed 256×256 input size.
- **Decoder:** 4 upsampling stages, each a **`ConvTranspose2d(kernel=2, stride=2)`** (learned upsampling) that halves channel count, followed by channel-wise concatenation with the matching encoder skip connection, then `DoubleConv`. A `bilinear=True` constructor flag exists as an alternative (fixed bilinear upsampling instead of a learned transposed convolution) but is not the default and was not used in verification.
- **Output head:** `Conv2d(base_channels, 1, kernel_size=1)` → 1×256×256.
- **Parameter count:** 31,037,633 (verified via `sum(p.numel() for p in model.parameters())`).

### 2.2 Output activation: raw logits, not Sigmoid
The model's final layer outputs **raw, unbounded logits** — no `Sigmoid` is applied inside the model. This is paired with `BCEDiceLoss` for the training objective (§3.2a) — a combined `BCEWithLogitsLoss` + soft Dice loss, both computed directly from logits. The `BCEWithLogitsLoss` half is a deliberate numerical-stability choice: it computes binary cross-entropy directly from logits using a log-sum-exp formulation (equivalent to `max(x,0) - x·z + log(1+exp(-|x|))`), which avoids the failure mode of a separate `Sigmoid` + `BCELoss` pipeline where `sigmoid(x)` can saturate to exactly `0.0` or `1.0` in floating point for large-magnitude logits, producing `log(0) = -inf` and NaN gradients. `torch.sigmoid()` is applied externally wherever an actual probability or binary mask is needed — at validation time (§3.3) and at inference time.

### 2.3 Verification (architecture only, no training)
A forward-pass smoke test (dummy input `[1, 3, 256, 256]` on the CUDA device) confirmed output shape `[1, 1, 256, 256]`, dtype `float32`, on `cuda:0`. This confirms architectural correctness only.

### 2.4 Model 2 — Attention U-Net (`models/segmentation/attention_unet.py`)
Same encoder, channel progression, and I/O contract as Model 1 (§2.1) — the encoder (`DoubleConv`/`Down`) is imported directly from `unet.py` rather than duplicated. The decoder differs: each skip connection is passed through an **additive attention gate** (Oktay et al., 2018) before concatenation. The gate projects the decoder's (already-upsampled) gating signal `g` and the encoder skip `x` into a shared intermediate channel space via two independent `1×1 Conv → BatchNorm` branches, sums them, applies `ReLU`, then a `1×1 Conv → BatchNorm → Sigmoid` to produce a per-pixel attention coefficient in `[0, 1]`, which rescales the skip connection (`x_gated = x · attention`) before it is concatenated with the upsampled decoder features.

Rationale specific to this dataset: since the segmentation mask (§1.2) covers a small, irregularly-shaped foreground within a mostly-black/padded 256×256 canvas, an attention mechanism that can learn to suppress background activations in the skip connections is a plausible way to improve on the standard U-Net's unweighted skip concatenation.

Verified via the same forward-pass smoke test as Model 1: output shape `[1, 1, 256, 256]`, `float32`, `cuda:0`. Parameter count: **31,389,165**.

### 2.5 Model 3 — ResUNet (`models/segmentation/resunet.py`)
Same overall topology and I/O contract as Model 1, but every `DoubleConv` block is replaced by a **residual block** (Zhang et al., 2017, "Road Extraction by Deep Residual U-Net"): two `3×3 Conv → BatchNorm` layers, with a shortcut connection — a `1×1 Conv → BatchNorm` projection (since every block in this network changes channel depth) — added to the second layer's output before the final `ReLU`. The intent is the standard residual-learning argument: each block learns a refinement relative to its input rather than a full transformation, which can ease gradient flow in deeper networks.

Verified via the same forward-pass smoke test: output shape `[1, 1, 256, 256]`, `float32`, `cuda:0`. Parameter count: **32,436,353**.

### 2.6 Model switching
All three models share an identical constructor signature (`Model(in_channels=3, out_channels=1)`) and forward contract (raw logits, `[B, 1, H, W]`). Training logic is not model- or dataset-specific: the shared engine (`scripts/trainer_engine.py`, §3) takes a model class/name *and* a dataset class as arguments (`dataset_cls`, defaulting to `ConjunctivaSegmentationDataset` for backward compatibility), and six thin entry-point scripts each just import the engine and call `run_study(model_cls=..., model_name=..., dataset_cls=...)` with their own architecture + dataset:

- `scripts/train_standard_unet.py`, `scripts/train_attention_unet.py`, `scripts/train_resunet.py` — original, crop-based `ConjunctivaSegmentationDataset` (§1.2), `model_name` = `unet`/`attention_unet`/`resunet`.
- `scripts/train_standard_unet_aligned.py`, `scripts/train_attention_unet_aligned.py`, `scripts/train_resunet_aligned.py` — raw-photo-aligned `AlignedConjunctivaSegmentationDataset` (§1.4), `model_name` = `unet_aligned`/`attention_unet_aligned`/`resunet_aligned`.

The `_aligned` suffix on `model_name` is deliberate, not cosmetic: every output path (`outputs/checkpoints/best_{model_name}.pth`, `outputs/logs/{model_name}_*`) is derived from `model_name`, so without a distinct name an aligned run would silently overwrite the existing crop-based checkpoints/logs. This replaced an earlier single-script `MODEL_REGISTRY`/`MODEL_NAME`-constant design — separate entry points were needed because execution happens on Kaggle, where editing a shared constant before each run isn't part of the workflow; a dedicated script per (model, dataset) pair can be run as-is.

---

## Phase 2 — Segmentation Training Procedure (`scripts/trainer_engine.py` + per-model entry points)

**Status: a local 5-trial smoke test was started for Model 1 (Standard U-Net) and produced real, directly-observed results for 4 of 5 trials before being interrupted (§3.5). A separate, externally-executed run has also been reported by the project author from Kaggle (§3.6) — that report is not independently verified by this codebase/session. Models 2 and 3 (§2.4, §2.5) have not been trained (locally or externally) — only their forward pass has been verified.**

### 3.1 Hyperparameter optimization via Optuna
Training and validation are wrapped in an `objective(trial)` function passed to an Optuna `Study`. The sampler is `optuna.samplers.TPESampler` — a **Tree-structured Parzen Estimator**, a sequential Bayesian-optimization method that models `P(hyperparameters | performance)` from completed trials to propose the next candidate, as opposed to (uninformed) random or grid search. It is instantiated explicitly (`seed=42`) for reproducibility, though it is also Optuna's default sampler when none is specified.

Two hyperparameters are tuned per trial, both log-uniform (appropriate for scale-free parameters that plausibly matter across orders of magnitude):
- `learning_rate ~ LogUniform(1e-5, 1e-2)`
- `weight_decay ~ LogUniform(1e-6, 1e-3)`

### 3.2 Per-trial training loop
Each trial constructs a fresh model and dataset (the classes/name passed into `run_study()` by whichever entry-point script called it, §2.6), `AdamW` optimizer (using the trial's sampled `learning_rate`/`weight_decay`), and the loss function described in §3.2a — no state is shared across trials. Training runs for up to **30 epochs**, with **early stopping** (patience = 5 epochs, evaluated on validation loss with no improvement threshold/min-delta) to abandon poorly-performing trials early rather than exhausting the full epoch budget.

### 3.2a Loss function fix: `BCEDiceLoss` (class-imbalance collapse)
**Problem observed:** training against the raw-photo-aligned dataset (§1.4), where per-patient foreground can be well under 1% of the 256×256 canvas, collapsed to an all-background prediction — `val_loss` kept decreasing while `val_dice` stayed pinned at exactly `0.0000`. Plain `BCEWithLogitsLoss` averages loss *per pixel*, so with foreground this rare, predicting all-background is already a low-loss, easy-to-reach local minimum: there's very little gradient pressure to bother getting a tiny minority of pixels right.

**Fix:** `criterion` is now `BCEDiceLoss` (`scripts/trainer_engine.py`) — `BCEWithLogitsLoss` averaged 50/50 with a soft (differentiable) Dice loss computed from `sigmoid(logits)` directly (not the hard-thresholded Dice used for the §3.3 metric, which can't be backpropagated through). Dice is a *ratio*, not a per-pixel average, so it stays scale-invariant to how small the true foreground is and keeps penalizing an all-background prediction heavily regardless.

**Verified quantitatively:** on a synthetic mask matching the real aligned dataset's sparsity (130/65536 foreground pixels, the same ratio measured for patient `India_057`), plain `BCEWithLogitsLoss` penalized an all-background prediction only 0.0198 more than a correct one (0.0199 vs. 0.0000458 — both already tiny in absolute terms). `BCEDiceLoss` penalized the same all-background prediction **0.504 more** than the correct one (0.510 vs. 0.0057) — a ~25× stronger gradient signal against the exact collapse mode observed.

**Scope of this change:** `trainer_engine.py` is shared by all six entry-point scripts (§2.6), so this affects future runs of the original crop-based scripts too, not just the `_aligned` ones — the crop-based dataset is less extremely imbalanced but not immune to the same failure mode. This means re-running `train_standard_unet.py` (etc.) today would **not** exactly reproduce the already-logged Kaggle results in `outputs/logs/{unet,attention_unet,resunet}_*` (§3.6), which were trained under plain `BCEWithLogitsLoss` — those logs remain a historical record of that specific run, not a currently-reproducible configuration.

### 3.3 Validation metrics: sigmoid-threshold, then Dice/IoU
Because the model outputs raw logits (§2.2), validation applies `torch.sigmoid(logits)` followed by thresholding at `0.5` to obtain a binary predicted mask, *before* computing spatial overlap metrics (the loss itself, `BCEDiceLoss` §3.2a, still consumes the raw logits directly — only the metrics operate on the thresholded binary mask). For each validation batch:

- **Dice coefficient:** `Dice = (2·|P∩G| + ε) / (|P| + |G| + ε)`
- **Intersection over Union:** `IoU = (|P∩G| + ε) / (|P∪G| + ε)`

(`P` = predicted binary mask, `G` = ground-truth binary mask, `ε = 1e-7` for numerical stability against empty masks.) Per-sample scores are averaged with weighting by batch size, giving a correct sample-weighted mean across the full validation set (not a naive mean-of-batch-means, which would slightly misweight a trailing partial batch).

The trial's objective value returned to Optuna is the **best (maximum) validation Dice observed across all epochs in that trial** — tracked independently of which epoch triggered early stopping, so it reflects the best checkpoint the trial reached rather than only its final or stopping epoch. The validation IoU from that *same* epoch (not an independently-tracked IoU maximum, which could otherwise come from a different epoch) is stored via `trial.set_user_attr("best_val_iou", ...)`, alongside `trial.set_user_attr("model_name", model_name)` (the name passed into `make_objective()` by the calling entry-point script), so both are retrievable from `study.best_trial.user_attrs` without re-parsing console logs.

### 3.4 Execution plan
`optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))` followed by `study.optimize(objective, n_trials=5)` — a 5-trial smoke test intended to validate the pipeline mechanics (timing, metric tracking, hyperparameter proposal behavior) before committing to a larger search budget.

### 3.4a Output persistence
Added specifically because training runs on a remote, ephemeral Kaggle session: anything not written to disk before the session ends is unrecoverable. `run_study()` now persists, per model:
- **`outputs/checkpoints/best_{model_name}.pth`** — the state dict of whichever epoch, across *all* trials in the study (not just the final trial), achieved the highest validation Dice. Tracked via a `best_overall_dice` value held in the `make_objective()` closure, which persists across every trial Optuna runs against that closure instance.
- **`outputs/logs/{model_name}_trials.csv`** — the full per-trial record (`study.trials_dataframe()`: params, value, user attributes, state, duration) for every trial in the study.
- **`outputs/logs/{model_name}_study_summary.json`** — a compact summary of the winning trial (number, Dice, IoU, hyperparameters, checkpoint path, UTC timestamp).

`outputs/checkpoints/` is `.gitignore`'d (large binaries, regenerable by retraining); `outputs/logs/` is not — those files are small and are meant to be committed as part of the experimental record.

### 3.5 Local execution (partial, interrupted)
This 5-trial smoke test was started locally against Model 1 (Standard U-Net) on the RTX 4050. The process was killed (session/agent teardown between conversation turns) partway through Trial 4, before the script's own final "best trial" summary printed. Because Python fully buffers stdout when it is redirected to a file (rather than a terminal), the per-epoch `print()` lines (train/val loss, Dice, IoU) for the in-progress run were lost when the process was killed — only Optuna's own trial-completion log lines survived, since Optuna's logger flushes independently of the script's buffered prints. The following is the complete, real data recovered from that run:

| Trial | Result | Val Dice | learning_rate | weight_decay |
|---|---|---|---|---|
| 0 | complete (best) | 0.9893 | 0.000133 | 0.000711 |
| 1 | complete | 0.9852 | 0.00157 | 0.0000625 |
| 2 | complete | 0.9833 | 0.0000294 | 0.0000029 |
| 3 | complete | 0.9733 | 0.0000149 | 0.000397 |
| 4 | **interrupted, never finished** | — | — | — |

Per-epoch IoU and train/val loss values for these trials are not recoverable (lost to output buffering, per above) — only the per-trial best Dice and hyperparameters, from Optuna's surviving log lines.

### 3.6 External execution report — Kaggle (user-reported, not independently verified)
**This section records a claim reported by the project author from a training run executed on Kaggle, outside this coding session. This session has not observed the underlying notebook, execution logs, or a model checkpoint for this run, and cannot independently verify the figures below. They are recorded here at the author's explicit request, attributed as such, for thesis note-taking purposes — not as a codebase-verified result.**

Reported infrastructure:
- Platform: Kaggle Notebooks, GPU **T4×2** — reported as a switch away from the P100, because PyTorch 2.10+ dropped support for the older `sm_60` GPU architecture.
- Reported data-handling change: the pre-processed dataset (including `dataset_splits.csv`) is physically copied into `/kaggle/working/.../data/processed/` before training, to avoid Kaggle's symlink/read-only filesystem restrictions.

Reported results (Model 1, Standard U-Net, 5-trial Optuna search):
- Best trial: **Trial 4** — `learning_rate ≈ 6×10⁻⁴`, `weight_decay ≈ 1×10⁻⁴`.
- Reported Validation Dice: **0.9900**; reported Validation IoU: **0.9800**.
- Reported train loss 0.0291 / val loss 0.0191 at epoch 30, described by the author as showing no overfitting.
- Author's conclusion: the Standard U-Net baseline is considered complete, with no further tuning planned for this model.

**Noted discrepancy:** the locally-observed run (§3.5), using the identical script and search space, produced a different best trial (Trial 0, Dice 0.9893) and never completed a Trial 4 at all. This does not prove the Kaggle report is incorrect — different hardware (T4×2 vs. RTX 4050) and Optuna's stochastic TPE proposals can legitimately produce different trial trajectories across separate runs/environments — but it means the two results cannot be reconciled from evidence available in this session. **For thesis purposes:** retain the actual Kaggle notebook output (exported cell output, a saved metrics/log file, or the notebook link) as a citable, checkable artifact before reporting the 0.9900/0.9800 figures as a confirmed thesis result.
