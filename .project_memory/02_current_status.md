# Current Status — EYES-DEFY-ANEMIA

Last updated: 2026-07-17

## What's working right now
- Full Phase 0 pipeline, reproducible from `archive.zip` (`scripts/phase0_prepare_dataset.py`).
- `scripts/dataset.py`: three dataset classes (`ConjunctivaSegmentationDataset`, `AlignedConjunctivaSegmentationDataset`, `AnemiaClassificationDataset`), stratified split builder, train/eval transforms, `get_dataloaders()`.
- `scripts/build_aligned_dataset.py`: verified end-to-end — 217/217 patients aligned, template-match confidence 0.952–0.998 (mean 0.992), zero dimension mismatches, confirmed both quantitatively (geometric pixel-count ratio check) and visually (5 spot-checked patients, both countries).
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
1. *(User, in progress)* Upload `data/processed/aligned_raw/` to Kaggle, update the Kaggle-side copy/setup script to place it correctly, `git pull` the latest code (now includes the `BCEDiceLoss` fix).
2. Run the 3 `_aligned` entry-point scripts on Kaggle *(re-run — the first attempt collapsed under the old plain-BCE loss)*.
3. Pull `outputs/checkpoints/best_*_aligned.pth` + `outputs/logs/*_aligned_*` back into this repo.
4. **Re-run a domain-shift check** — does the aligned-trained model actually isolate tissue on a raw photo now? Don't skip this; it's the entire point of the pivot.
5. Only after that: rebuild Phase 3 (real inference + cropping) against whichever model is confirmed to generalize.
6. Phase 4 (classification) hasn't been started at all yet.
