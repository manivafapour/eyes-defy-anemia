# Project Roadmap — EYES-DEFY-ANEMIA

High-level, sequential plan for the whole project. `[x]` = done and verified (not just written). Detailed methodology/evidence for each verified step lives in `CLAUDE.md`; this file is just the shape of the plan.

## Phase 0 — Data Engineering
- [x] Unify India + Italy metadata, derive WHO-threshold anemia labels
- [x] Extract + standardize raw photos and palpebral crops (pad-to-square, 256×256, Lanczos)
- [x] Repair corrupted PNG chunks (`iCCP` CRC issue, 63/217 files)
- [x] Identify demographic bias (India vs. Italy anemia rates, gender composition skew)

## Phase 1 — Data Pipeline Construction
- [x] Patient-level, `country + anemic_label`-stratified 70/15/15 split
- [x] `AnemiaClassificationDataset`. (A crop-based `ConjunctivaSegmentationDataset` also existed here — removed 2026-08-08 along with the 3 hand-built models that used it, per the project author's explicit request, once the pretrained-architecture sweep superseded them. See `04_pretrained_architecture_sweep.md`.)
- [x] Synchronized `albumentations` augmentation pipeline (train vs. eval transforms)
- [x] **Data-centric fix (v1, template matching — proven wrong, discarded):** see `CLAUDE.md` §1.4.1
- [x] **Data-centric fix (v2, SIFT/ORB + RANSAC homography):** 201/217 aligned (originally 202/217, corrected after fixing a white-background mask bug — `CLAUDE.md` §1.4.4), visually confirmed (`India_071` + spot-checks). The remaining 16 are permanently excluded (not manually annotated — tried and rejected, `CLAUDE.md` §1.4.3); `AlignedConjunctivaSegmentationDataset` filters to only these 201 without touching the shared `dataset_splits.csv`.

## Environment / Hardware
- [x] Local CUDA-enabled PyTorch environment (RTX 4050)
- [x] Kaggle T4×2 workflow established (external training, results pulled back manually)

## Phase 2 — Segmentation Modeling
- [x] ~~Standard U-Net, Attention U-Net, ResUNet architectures~~ — **removed from the codebase 2026-08-08** (project author's explicit request), superseded by the pretrained-architecture sweep below. `CLAUDE.md` §2.1-2.5 keeps the historical architecture spec/results; the code, entry-point scripts, and Kaggle checkpoints/logs no longer exist in the repo. See `04_pretrained_architecture_sweep.md` for the full removal record.
- [x] Optuna training engine (`trainer_engine.py`) — kept and extended, not removed; now defaults to `AlignedConjunctivaSegmentationDataset` since the crop-based dataset is gone.

### Phase 2, original scope (historical — superseded, kept for context)
- [x] Trained on the ORIGINAL crop-based dataset via Kaggle, all 3 models (results were logged — see `CLAUDE.md` §3.6; the log files themselves were deleted with the rest of this implementation on 2026-08-08)
- [ ] ~~Retrain all 3 models on the ALIGNED raw-photo dataset~~ — moot now that those 3 models no longer exist; superseded by the pretrained sweep below, which trains on the aligned dataset(s) directly
- [ ] Verify a generalizing model actually isolates tissue on a raw photo — still an open item, now to be answered by whichever model wins the pretrained sweep, not the old 3

### Phase 2 expansion: 9-architecture pretrained sweep × 2 tissue types (started 2026-08-08)
Mirrors classification's "9 architectures, compare on metrics" structure, adapted to segmentation: 3 CNN + 3 Hybrid (CNN+Transformer) + 3 Pure Transformer at increasing parameter tiers, all pretrained (ImageNet encoder-only for CNN/Hybrid; full ADE20K encoder+decoder for the 3 Transformer entries), fine-tuned on both tissue types. Full detail in `04_pretrained_architecture_sweep.md`.
- [x] **New data engineering: `forniceal_palpebral` aligned segmentation dataset** (`scripts/build_aligned_dataset_forniceal.py`) — did not exist before this session; reuses the palpebral pipeline's SIFT/ORB+RANSAC algorithm unchanged, pointed at the second crop. **211/217 aligned, 0 genuine alignment failures** (6 excluded for having no forniceal crop at all — matches classification's independently-documented list exactly). Area ratio to raw photo is tightly consistent (13.73–14.03×) — strong evidence of correct geometry. Visually spot-checked (4 patients incl. weakest-inlier and white-bg cases) — all correct. New sanity-check notebook (`notebooks/verify_forniceal_alignment_sanity_check.ipynb`) built and executed clean (0 blank/near-blank masks) — **full human visual review still recommended before Kaggle training**, same gate as every prior alignment pipeline in this project.
- [x] `AlignedConjunctivaSegmentationDataset` generalized with a `tissue_type` param (`"palpebral"` default = unchanged behavior, `"forniceal_palpebral"` = new) rather than duplicated into a second class.
- [x] `trainer_engine.py` extended (`build_model`, `image_size`, `tissue_type` params on `make_objective`/`run_study`) — verified via import + signature check. (Originally additive alongside the 3 hand-built models' 6 entry-point scripts; those were removed 2026-08-08 per the project author, so `trainer_engine.py`'s default `dataset_cls` now points at `AlignedConjunctivaSegmentationDataset` instead of the deleted crop-based class.)
- [x] New dependencies installed and verified: `segmentation-models-pytorch` 0.5.0, `timm` 1.0.28, `transformers` 5.14.1, `einops` 0.8.2 — all compatible with Python 3.14.6 / torch 2.13.0+cu130.
- [x] 9-model `ARCHITECTURE_REGISTRY` built (`models/segmentation/pretrained_registry.py`) + custom `TransUNet` (`models/segmentation/transunet.py`, composed from pretrained `timm` ResNet50 + ViT-B/16 rather than an unmaintained third-party package). All 9 structurally verified (forward pass + real param count) — see `04_pretrained_architecture_sweep.md` for the full table.
- [x] 18 entry-point scripts generated (`scripts/train_pretrained/`, 9 architectures × 2 tissue types), all `py_compile`-verified.
- [x] **18-combo × 12-trial Kaggle sweep — 12/18 completed.** All 6 CNN + Hybrid combos (both tissue types) trained successfully; all 6 Transformer combos (SegFormer-B2, Swin-Base+UperNet, Swin-Large+UperNet) OOM'd on Trial 0's first batch (`input_size=512` too large for the T4 at the shared `BATCH_SIZE=16`) — deterministic, not a fluke, root-caused from the real Kaggle traceback (`.project_memory/kaggle/01_kaggle_notes.md`). Fixing and re-running those 6 was explicitly ruled out (time constraint). Results ranked in `Segmentation/scripts/train_pretrained/output/model_ranking_v2.xlsx` — best: TransUNet (Hybrid, Strong) on forniceal_palpebral, Test Dice 0.930.
- [ ] **In progress: fixed-hyperparameter 3-fold re-training of the 12 completed combos** (`scripts/train_pretrained_kfold/`, 3 tier Kaggle notebooks — Base/Mid/Strong, 4 combos each) — full rationale, design decisions, and verification record in `.project_memory/05_kfold_reevaluation.md`. Infrastructure built and verified (structural checks + a real end-to-end dry run); no real Kaggle run has happened yet.

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

## Repository organization
- [x] Moved the whole segmentation-phase tree into a dedicated `Segmentation/` top-level folder (2026-07-29), sibling to `classification/`, for a clean repo structure. `CLAUDE.md` stays at the repo root (reasoning in `03_tech_stack_and_rules.md`) but all its internal paths were updated. Full detail in `02_current_status.md`.
