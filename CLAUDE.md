# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It also serves as the primary methodology reference for the project's written thesis — entries below are written for scientific accuracy and should be treated as authoritative over any prior/looser description of the same step.

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

---

## Computational Environment / Hardware Setup

- GPU: **NVIDIA GeForce RTX 4050 Laptop GPU** (driver reports CUDA capability up to 13.1 via `nvidia-smi`).
- An initial unpinned `pip install torch` resolved to a **CPU-only** build (`torch 2.13.0+cpu`) — no GPU acceleration.
- This was superseded by an explicit CUDA-enabled reinstall: **`torch 2.13.0+cu130`**, **`torchvision 0.28.0+cu130`**, **`torchaudio 2.11.0+cu130`** (built against the CUDA 13.0 runtime, which the 13.1-capable driver runs without issue). `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` both confirmed against the RTX 4050 in this environment.
- Python: 3.14.6, project virtual environment at `venv/`.
- Other pinned/verified library versions in this environment: pandas 3.0.3, numpy 2.5.1, Pillow 12.3.0, openpyxl 3.1.5, albumentations 2.0.8, opencv-python (`cv2`) 5.0.0, scikit-learn 1.9.0, optuna 4.9.0.
- `requirements.txt` currently lists only the Phase 0 dependencies (`pandas`, `openpyxl`, `Pillow`) and has not yet been updated to include the Phase 1/2 dependencies above.

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
The model's final layer outputs **raw, unbounded logits** — no `Sigmoid` is applied inside the model. This is paired with `torch.nn.BCEWithLogitsLoss` for the training objective (§3), a deliberate numerical-stability choice: `BCEWithLogitsLoss` computes binary cross-entropy directly from logits using a log-sum-exp formulation (equivalent to `max(x,0) - x·z + log(1+exp(-|x|))`), which avoids the failure mode of a separate `Sigmoid` + `BCELoss` pipeline where `sigmoid(x)` can saturate to exactly `0.0` or `1.0` in floating point for large-magnitude logits, producing `log(0) = -inf` and NaN gradients. `torch.sigmoid()` is applied externally wherever an actual probability or binary mask is needed — at validation time (§3) and at inference time.

### 2.3 Verification (architecture only, no training)
A forward-pass smoke test (dummy input `[1, 3, 256, 256]` on the CUDA device) confirmed output shape `[1, 1, 256, 256]`, dtype `float32`, on `cuda:0`. This confirms architectural correctness only — no training has occurred and no accuracy/Dice/IoU results exist yet for this model.

---

## Phase 2 — Segmentation Training Procedure (`scripts/train_segmentation.py`)

**Status: written and structurally verified (imports resolve, syntax compiles), but not yet executed. No training has run; no empirical results exist for this component.**

### 3.1 Hyperparameter optimization via Optuna
Training and validation are wrapped in an `objective(trial)` function passed to an Optuna `Study`. The sampler is `optuna.samplers.TPESampler` — a **Tree-structured Parzen Estimator**, a sequential Bayesian-optimization method that models `P(hyperparameters | performance)` from completed trials to propose the next candidate, as opposed to (uninformed) random or grid search. It is instantiated explicitly (`seed=42`) for reproducibility, though it is also Optuna's default sampler when none is specified.

Two hyperparameters are tuned per trial, both log-uniform (appropriate for scale-free parameters that plausibly matter across orders of magnitude):
- `learning_rate ~ LogUniform(1e-5, 1e-2)`
- `weight_decay ~ LogUniform(1e-6, 1e-3)`

### 3.2 Per-trial training loop
Each trial constructs a fresh `UNet`, `AdamW` optimizer (using the trial's sampled `learning_rate`/`weight_decay`), and `BCEWithLogitsLoss` — no state is shared across trials. Training runs for up to **30 epochs**, with **early stopping** (patience = 5 epochs, evaluated on validation loss with no improvement threshold/min-delta) to abandon poorly-performing trials early rather than exhausting the full epoch budget.

### 3.3 Validation metrics: sigmoid-threshold, then Dice/IoU
Because the model outputs raw logits (§2.2), validation applies `torch.sigmoid(logits)` followed by thresholding at `0.5` to obtain a binary predicted mask, *before* computing spatial overlap metrics (the loss itself, `BCEWithLogitsLoss`, still consumes the raw logits directly — only the metrics operate on the thresholded binary mask). For each validation batch:

- **Dice coefficient:** `Dice = (2·|P∩G| + ε) / (|P| + |G| + ε)`
- **Intersection over Union:** `IoU = (|P∩G| + ε) / (|P∪G| + ε)`

(`P` = predicted binary mask, `G` = ground-truth binary mask, `ε = 1e-7` for numerical stability against empty masks.) Per-sample scores are averaged with weighting by batch size, giving a correct sample-weighted mean across the full validation set (not a naive mean-of-batch-means, which would slightly misweight a trailing partial batch).

The trial's objective value returned to Optuna is the **best (maximum) validation Dice observed across all epochs in that trial** — tracked independently of which epoch triggered early stopping, so it reflects the best checkpoint the trial reached rather than only its final or stopping epoch.

### 3.4 Execution plan (not yet run)
`optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))` followed by `study.optimize(objective, n_trials=5)` — a 5-trial smoke test intended to validate the pipeline mechanics (timing, metric tracking, hyperparameter proposal behavior) before committing to a larger search budget.
