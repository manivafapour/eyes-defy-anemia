# Current Status — EYES-DEFY-ANEMIA

Last updated: 2026-07-18

## Where things stand
The aligned-photo segmentation dataset went through two major corrections this session and is now considered solid:

1. **v1 alignment (`cv2.matchTemplate`) was proven wrong.** A manual visual review of `India_071` found its mask floating on the sclera instead of the lower eyelid, despite a 0.995 confidence score — appearance correlation doesn't guarantee correct location. Discarded entirely.
2. **v2 alignment (SIFT/ORB + `cv2.findHomography(..., RANSAC)`)** replaced it — geometrically constrained, not a single appearance score. Result: 202/217 aligned, 15 honest failures (rejected by geometric sanity checks, not silently wrong). Visually confirmed correct on `India_071` + several spot-checks. Full writeup: `CLAUDE.md` §1.4.1 (v1) / §1.4.2 (v2).
3. **The 15 failures are permanently excluded**, not manually fixed. A `cv2.selectROI` manual-bbox tool was built and actually run on all 15, but the project author rejected the approach (crude/inconsistent vs. the other 202's RANSAC-derived masks) — tool and its output discarded. `AlignedConjunctivaSegmentationDataset` now filters to `status == "ok"` (202) without touching the shared `dataset_splits.csv` (`ConjunctivaSegmentationDataset`/`AnemiaClassificationDataset` still see all 217). Verified: `aligned_seg_*` returns 143/28/31 (202); `seg_*`/`cls_*` unaffected at 151/33/33 (217). Side effect: exclusion skews India-heavy (13 India, 2 Italy dropped). Full detail: `CLAUDE.md` §1.4.3.

**Training is authorized to proceed on this 202-patient aligned dataset.** The CLAUDE.md §1.4.2 gate ("no training until the project author has personally reviewed `verify_alignment_sanity_check.ipynb` and confirmed it") was explicitly cleared by the author on 2026-07-18 — recorded here since the review-confirmation event itself wasn't previously logged anywhere checkable.

## The collapse mystery — two fixes applied, root cause still not 100% certain
An early Kaggle training attempt on the (since-discarded, v1) aligned dataset collapsed to all-background (`val_dice` pinned at `0.0000`, `val_loss` stuck ~0.60–0.61). Two things happened in response:
- **Fix 1 — `BCEDiceLoss`** (`CLAUDE.md` §3.2a): replaced plain `BCEWithLogitsLoss`, whose per-pixel averaging under-penalizes a rare foreground. Verified: penalizes an all-background collapse 25× harder than plain BCE did, on synthetic data matching the real sparsity.
- **Re-interpretation:** the collapsed run was against v1's *wrong* masks. A model trained on non-blank-but-spatially-wrong targets would plausibly collapse regardless of loss function (no learnable image→mask relationship). So it's genuinely unknown whether `BCEDiceLoss` alone would have fixed it, or whether the real problem was always the bad masks.
- **Fix 2 (this session) — `FocalTverskyLoss` + side-by-side comparison** (`CLAUDE.md` §3.2b): rather than assume `BCEDiceLoss` is sufficient, `loss_fn` is now itself an Optuna-tuned categorical hyperparameter (`bce_dice` vs. `focal_tversky`), so the next Kaggle run will directly compare both on the clean, 202-patient v2 data. `FocalTverskyLoss` (α=0.3, β=0.7, γ=4/3) targets the same failure mode more aggressively — verified to penalize an all-background collapse even harder than `BCEDiceLoss` (0.9999 vs. 0.0013, vs. `BCEDiceLoss`'s 0.510 vs. 0.0057) and to punish a 50%-recall partial miss substantially more too (0.311 vs. 0.177).
- Real, current sparsity stats on the 202-patient set: median 4.15% foreground, but bimodal — a small Italy-crop cluster sits at 75%. This is why a global BCE `pos_weight` was rejected in favor of Tversky-family losses (computed per-sample as a ratio, so they adapt automatically).

**This has not been run on Kaggle yet.** Nothing is confirmed until real training data comes back.

## What's working right now
- Full Phase 0 pipeline, reproducible from `archive.zip` (`scripts/phase0_prepare_dataset.py`).
- `scripts/dataset.py`: three dataset classes, stratified split builder, transforms, `get_dataloaders()`. `AlignedConjunctivaSegmentationDataset` correctly filters to the 202 aligned patients (no more `FileNotFoundError` gap).
- `scripts/build_aligned_dataset.py` (v2, SIFT/ORB + homography): 202/217 aligned, 15 honest failures, inliers 5–2014 (mean 311).
- Three segmentation architectures (`models/segmentation/{unet,attention_unet,resunet}.py`), forward-pass verified on GPU.
- `scripts/trainer_engine.py`: model-, dataset-, *and now loss-function*-agnostic Optuna engine. `loss_fn` sampled per-trial from `LOSS_REGISTRY` (`bce_dice`, `focal_tversky`); each gets its own checkpoint (`best_{model_name}_{loss_fn}.pth`) plus a per-loss comparison table in the study summary JSON. Verified via a synthetic 6-trial run with mocked results.
- Six entry-point scripts (3 original crop-based + 3 `_aligned`) — no changes needed for the loss comparison, since loss selection lives entirely inside the shared engine now.

## Training results that exist right now
**Original crop-based dataset, all 3 models trained on Kaggle T4×2 (before the loss-fn-as-hyperparameter change, under plain `BCEDiceLoss`):**
- Real per-trial logs: `outputs/logs/{unet,attention_unet,resunet}_trials.csv` + `_study_summary.json`.
- Standard U-Net: a Kaggle result (Trial 4, Val Dice 0.9900, Val IoU 0.9800) was reported but **not independently verified** — no notebook/log artifact available. A local partial run showed different numbers (Trial 0 best, Dice 0.9893). Both recorded in `CLAUDE.md` §3.5/§3.6 with explicit provenance labels — don't cite the 0.99/0.98 figures as confirmed.

**Aligned dataset (v2, 202 patients, with the loss comparison):** no training has been run yet.

## 2026-07-18 session: pre-flight check before the real Kaggle run
Before triggering the actual Kaggle training, this session verified the state left by the previous one and found one real bug:
- **`data/processed/aligned_raw.zip` (sitting in the working dir, untracked) was stale v1 data** — its `alignment_log.csv` had the old template-matching schema (`status,confidence,match_x,match_y,...`, 217/217 "ok"), i.e. the *discarded, proven-wrong* dataset (sclera-mask bug, §1.4.1), not the current 202-patient v2 SIFT/RANSAC data. Had this been uploaded to Kaggle as-is, it would have silently resurrected the exact bug that was already found and fixed. **Regenerated correctly** from the current `data/processed/aligned_raw/` directory on disk (which was already correct — v2 schema, 202 ok / 15 failed) — verified the new zip's `alignment_log.csv` now shows the SIFT-schema columns and 202 images/masks.
- `n_trials` was confirmed still at the engine default of 5, not bumped to 10-12 as `.project_memory` had only *recommended* (never implemented) — matches the project author's own recollection. **Decision: the three `_aligned` entry-point scripts now explicitly pass `n_trials=12`** (`run_study(..., n_trials=12)`), overriding the shared engine's 5-trial default. The three original crop-based scripts are untouched (still implicitly 5, matching their already-logged §3.5/§3.6 results). See `CLAUDE.md` §3.4 for the rationale (loss_fn as a 3rd tuned dimension needs more trials for fair bce_dice/focal_tversky coverage).
- `my_valuable_outputs.zip` (351MB, untracked) confirmed as a backup of the original crop-based Kaggle run's checkpoints/logs (§3.6) — fine as-is, no action taken.
- No Kaggle CLI is installed/configured in this environment, and there's no in-repo Kaggle setup/upload script — the actual upload to Kaggle and running of the 3 `_aligned` notebooks remains a manual, external step for the project author.

## Immediate next step
1. **Project author uploads the corrected `data/processed/aligned_raw.zip` to Kaggle** (new dataset version — the old one there is the stale v1 data, same bug as the local zip had). `git pull` the latest code (includes the `n_trials=12` change).
2. Run the 3 `_aligned` entry-point scripts on Kaggle (now 12 trials each, per-loss-function comparison built in).
3. Pull `outputs/checkpoints/best_*_aligned*.pth` + `outputs/logs/*_aligned_*` back into this repo.
4. Compare `bce_dice` vs. `focal_tversky` via the per-loss comparison table in the study summary JSON.
5. **Re-run a domain-shift check** — does the winning model actually isolate tissue on a raw photo? This is still the entire point of the original pivot; don't skip it.
6. Only after that: rebuild Phase 3 (real inference + cropping) against whichever model generalizes.
7. Phase 4 (classification) hasn't been started at all yet.
