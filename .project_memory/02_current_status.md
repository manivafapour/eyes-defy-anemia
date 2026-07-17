# Current Status — EYES-DEFY-ANEMIA

Last updated: 2026-07-17

## CRITICAL: the aligned_raw dataset was rebuilt from scratch — v1 was wrong
The original `scripts/build_aligned_dataset.py` (OpenCV `matchTemplate`, "217/217 aligned, confidence 0.952–0.998") **was proven wrong**. A manual visual review of `India_071` (not one of the 5 patients originally spot-checked) found the mask floating on the sclera instead of the lower eyelid, despite a 0.995 confidence score. Lesson: appearance-correlation confidence does not imply correct location, and 5 hand-picked spot-checks were not enough coverage to catch this. **The entire v1 approach and its `aligned_raw` output were discarded.**

Rebuilt (v2) using SIFT/ORB feature matching + `cv2.findHomography(..., cv2.RANSAC)` — geometrically constrained, not a single appearance score. Full writeup: `CLAUDE.md` §1.4.1 (v1, why it failed) and §1.4.2 (v2, current method). Result: 202/217 aligned, 15 *honest* failures (rejected by geometric sanity checks rather than silently wrong), all via SIFT, inliers 5–2014. `India_071` now visually confirmed correct, along with several other spot-checks.

**Status: pending the project author's own visual review of `notebooks/verify_alignment_sanity_check.ipynb` (v2) before any training proceeds.** Do not resume training on `aligned_raw` until that happens.

## What's working right now
- Full Phase 0 pipeline, reproducible from `archive.zip` (`scripts/phase0_prepare_dataset.py`).
- `scripts/dataset.py`: three dataset classes (`ConjunctivaSegmentationDataset`, `AlignedConjunctivaSegmentationDataset`, `AnemiaClassificationDataset`), stratified split builder, train/eval transforms, `get_dataloaders()`. **Known gap:** `AlignedConjunctivaSegmentationDataset` will raise `FileNotFoundError` for the 15 patients missing from the v2 `aligned_raw` output, since `dataset_splits.csv` still lists all 217 — not yet handled, since no training against this dataset is authorized yet.
- `scripts/build_aligned_dataset.py` (v2, SIFT/ORB + homography): 202/217 patients aligned, 15 honest failures, inlier counts 5–2014 (mean 311), 0/202 produced masks blank/near-blank. Verified quantitatively (scale-ratio consistency check) and visually (`India_071`, `India_001`, `India_029`, `Italy_097`, `Italy_001`).
- Three segmentation architectures (`models/segmentation/{unet,attention_unet,resunet}.py`), all forward-pass verified on GPU.
- `scripts/trainer_engine.py`: model- *and* dataset-agnostic Optuna training engine (TPE sampler, early stopping, Dice/IoU, checkpoint + log persistence to `outputs/`).
- Six entry-point scripts total: 3 original (crop-based dataset) + 3 `_aligned` (raw-photo-aligned dataset). **Only the original 3 have actually been trained** (via Kaggle) — the `_aligned` scripts are written and import-verified but have never been run.

## Training results that exist right now
**Original crop-based dataset, all 3 models trained on Kaggle T4×2:**
- Real per-trial logs exist: `outputs/logs/{unet,attention_unet,resunet}_trials.csv` + `_study_summary.json`.
- Standard U-Net specifically: the project author reported a Kaggle result (Trial 4, Val Dice 0.9900, Val IoU 0.9800) that is **not independently verified** in this session — no notebook/log artifact was available to check it against. A separate local partial run (interrupted mid-Trial-4 by session teardown) produced different, directly-observed numbers (Trial 0 best, Dice 0.9893). Both figures are recorded in `CLAUDE.md` §3.5/§3.6, explicitly labeled by provenance — don't quote the 0.99/0.98 figures as confirmed without going back to check for the actual Kaggle notebook output.

**Aligned dataset:** an initial Kaggle training attempt on the aligned dataset **collapsed** — model predicted all-background (`val_dice` pinned at `0.0000`) while `val_loss` kept decreasing. Root cause + fix below; not yet re-run with the fix.

## Known issues driving the current direction
1. **Domain shift (led to the aligned-dataset pivot):** the original `ConjunctivaSegmentationDataset` pairs an image that's already ~90%+ black outside the true mask (its "image" is the palpebral crop's own RGB, zeroed wherever its own alpha channel is 0 — `CLAUDE.md` §1.2). A model trained on this does not generalize to raw photos: empirically verified in the abandoned Phase 3 attempt 1 — 214/217 outputs came out near-full-frame, and visual inspection confirmed the model wasn't isolating tissue, just predicting "foreground = non-black pixel." This is why we pivoted to building `aligned_raw` (a real mask in the raw photo's own coordinate frame, via template matching) instead of patching the old approach.
2. **Class-imbalance collapse (found when first training on the aligned dataset):** per-patient foreground in `aligned_raw` masks can be well under 1% of the 256×256 canvas. Plain `BCEWithLogitsLoss` let the model minimize loss by predicting all-background. **Fixed** in `scripts/trainer_engine.py`: `criterion` is now `BCEDiceLoss` (BCE + soft Dice, 50/50). Verified quantitatively on a synthetic mask matching the real sparsity: plain BCE penalized an all-background collapse by only 0.0198 over a correct prediction; `BCEDiceLoss` penalizes the same collapse by 0.504 — ~25× stronger gradient signal. Full writeup: `CLAUDE.md` §3.2a. **This change affects all six entry-point scripts** (shared engine), so re-running the original (non-aligned) scripts today would no longer exactly reproduce the already-logged Kaggle results in `outputs/logs/{unet,attention_unet,resunet}_*`, which were trained under plain BCE.

## Immediate next step
1. **User visually reviews `notebooks/verify_alignment_sanity_check.ipynb` (v2)** — all 10 sampled patients (including `India_071`) plus the full-dataset failure list. Nothing below happens until this is confirmed.
2. Decide how to handle the 15 patients whose alignment failed (drop from the split? attempt a different method for just those?) and fix `AlignedConjunctivaSegmentationDataset`'s missing-file gap accordingly.
3. Re-upload the new `aligned_raw/` to Kaggle (the old upload is now stale/wrong), update the Kaggle-side copy/setup script, `git pull` the latest code.
4. Run the 3 `_aligned` entry-point scripts on Kaggle (this will be a fresh run against genuinely-correct data, with the `BCEDiceLoss` fix already in place).
5. Pull `outputs/checkpoints/best_*_aligned.pth` + `outputs/logs/*_aligned_*` back into this repo.
6. **Re-run a domain-shift check** — does the aligned-trained model actually isolate tissue on a raw photo now? Don't skip this; it's the entire point of the pivot.
7. Only after that: rebuild Phase 3 (real inference + cropping) against whichever model is confirmed to generalize.
8. Phase 4 (classification) hasn't been started at all yet.

## Update: training still collapsed after the BCEDiceLoss fix
Kaggle report: `val_dice` still pinned at `0.0000`, `val_loss` stuck ~0.60–0.61, even with `BCEDiceLoss` in place. Before touching the loss/hyperparameters further, built `notebooks/verify_alignment_sanity_check.ipynb` to check whether `aligned_raw` masks are actually blank.

**Result (executed against the local repo's data):** masks are **not** blank. Full scan of all 217 `aligned_raw` masks: 0 fully blank, 0 near-blank (<10 positive px), min positive pixels = 186, mean = 1094. Visual check on a fresh random patient (`Italy_069`, not previously hand-picked) shows the mask precisely on the correct tissue location, matching the original crop's shape.

**Conclusion at the time:** the data in *this local repo* is confirmed good — this specific collapse is not explained by blank/corrupted masks here. Two hypotheses were left open: (a) a Kaggle-side data mismatch, (b) the loss fix not being strong enough for the sparsity level.

**Superseded — likely real explanation found:** the "blank mask" check only checked for blank-ness, not spatial *correctness*. We now know (see the top of this file) that the v1 alignment approach producing that data was systematically capable of placing a fully non-blank, confident-looking mask at the *wrong anatomical location* (confirmed on `India_071`). Training a segmentation model against masks that are present but not consistently aligned with real tissue would plausibly cause exactly this collapse pattern (no learnable image→mask relationship, `val_loss` stuck at a middling value) independent of whether `BCEDiceLoss` was strong enough. **The class-imbalance loss fix (`BCEDiceLoss`) may well have been fine all along** — it was very possibly being asked to learn from corrupted training targets. This won't be known for certain until the v2-aligned data is used for a fresh training run.
