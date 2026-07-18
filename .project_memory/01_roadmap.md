# Project Roadmap — EYES-DEFY-ANEMIA

High-level, sequential plan for the whole project. `[x]` = done and verified (not just written). Detailed methodology/evidence for each verified step lives in `CLAUDE.md`; this file is just the shape of the plan.

## Phase 0 — Data Engineering
- [x] Unify India + Italy metadata, derive WHO-threshold anemia labels
- [x] Extract + standardize raw photos and palpebral crops (pad-to-square, 256×256, Lanczos)
- [x] Repair corrupted PNG chunks (`iCCP` CRC issue, 63/217 files)
- [x] Identify demographic bias (India vs. Italy anemia rates, gender composition skew)

## Phase 1 — Data Pipeline Construction
- [x] Patient-level, `country + anemic_label`-stratified 70/15/15 split
- [x] `ConjunctivaSegmentationDataset` (crop-based) + `AnemiaClassificationDataset`
- [x] Synchronized `albumentations` augmentation pipeline (train vs. eval transforms)
- [x] **Data-centric fix (v1, template matching — proven wrong, discarded):** see `CLAUDE.md` §1.4.1
- [x] **Data-centric fix (v2, SIFT/ORB + RANSAC homography):** 202/217 aligned, visually confirmed (`India_071` + spot-checks). The remaining 15 are permanently excluded (not manually annotated — tried and rejected, `CLAUDE.md` §1.4.3); `AlignedConjunctivaSegmentationDataset` filters to only these 202 without touching the shared `dataset_splits.csv`.

## Environment / Hardware
- [x] Local CUDA-enabled PyTorch environment (RTX 4050)
- [x] Kaggle T4×2 workflow established (external training, results pulled back manually)

## Phase 2 — Segmentation Modeling
- [x] Standard U-Net architecture
- [x] Attention U-Net architecture
- [x] ResUNet architecture
- [x] Optuna training engine (5-trial TPE search, early stopping, Dice/IoU, checkpoint + log persistence)
- [x] Model-switching *and* dataset-switching entry-point scripts
- [x] Trained on the ORIGINAL crop-based dataset via Kaggle, all 3 models (results logged — see `02_current_status.md` for what's verified vs. user-reported)
- [ ] Retrain all 3 models on the ALIGNED raw-photo dataset (202 patients) — infrastructure ready, including a `bce_dice` vs. `focal_tversky` side-by-side loss comparison (Optuna-tuned `loss_fn` categorical, `CLAUDE.md` §3.2b), now with `n_trials=12` for adequate per-loss coverage — not yet run (upload of corrected `aligned_raw.zip` to Kaggle is the pending manual step)
- [ ] Verify the aligned-trained model actually generalizes to raw photos (repeat the domain-shift check that failed before)

## Phase 3 — Tissue Isolation / Cropping
- [x] *(Attempt 1, abandoned)* Model-based inference cropping on raw photos — failed empirically (domain shift confirmed both quantitatively and visually), scripts deleted
- [x] *(Attempt 2, superseded)* Ground-truth-mask-based cropping — worked correctly, but superseded by fixing the root cause instead; scripts deleted
- [ ] **Real Phase 3 (pending):** once a model trained on `aligned_raw` is confirmed to generalize, rebuild the inference + cropping script against it

## Phase 4 — Anemia Classification (not started)
- [ ] Design/choose a classification architecture
- [ ] Training loop (loss, metrics — likely needs to account for the India/Italy class imbalance, §0.5 in `CLAUDE.md`)
- [ ] Decide: train on raw photos, Phase-3-cropped tissue, or compare both
- [ ] Evaluation against the held-out test split

## Final Deliverables
- [ ] Consolidated results/comparison across all segmentation + classification models
- [ ] Thesis writeup (`CLAUDE.md` is the running, cite-able methodology reference)
