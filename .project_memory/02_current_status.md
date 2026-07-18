# Current Status — EYES-DEFY-ANEMIA

Last updated: 2026-07-18

## ✅ RESOLVED: white-background mask bug fixed, `aligned_raw/` regenerated (201/217, was 202/217)
Full account: `CLAUDE.md` §1.4.4. The Kaggle run in flight against the buggy 202-patient data was manually stopped by the project author before this landed, so no contaminated training results exist to worry about.

- **What was wrong:** 30 patients (all Italy) had a GT mask covering ~75% of the frame — the whole eye, not the palpebral conjunctiva — because their source crop PNGs use an opaque white background instead of alpha transparency, and the old `(alpha > 127)` logic silently treated that as 100% foreground.
- **Fix:** `scripts/build_aligned_dataset.py` now auto-detects a non-functional alpha channel (`_alpha_is_functional()`, threshold 0.99 opaque fraction — real cutout patients top out at 13%, affected ones sit at exactly 100%, wide safety margin) and falls back to a `NOT near-white RGB` mask (`_white_background_mask()`, threshold 245) for both SIFT keypoint restriction and the final warp. `find_homography_alignment` itself (the actual SIFT/RANSAC logic) was not touched.
- **Full regeneration ran successfully:** 201/217 aligned (172 alpha + 29 white_bg fallback), 16 failed (was 15) — the one net change is `Italy_026`, which was a **false-positive "ok"** under the old permissive mask (73.41% coverage, weakest of the 30) and now honestly fails its geometric sanity check with the corrected, properly-restricted keypoint search. This is the intended "honest failure over silently-wrong success" behavior, not a regression. Mask coverage across all 201 is now a clean 1.78%–9.77% (mean 4.05%), no outliers, no bimodal cluster.
- **Caught and fixed a second, related bug during regeneration:** `process_patient()` only ever *writes* files for patients that succeed — it never deletes a previous run's output for a patient that used to succeed but now fails. This left `Italy_026`'s stale, wrong-mask files orphaned on disk after the fix (undetected until an explicit file-count cross-check against `alignment_log.csv`). Manually cleaned up for this run, and `main()` now auto-deletes orphaned output files on every future run (`CLAUDE.md`/script comments explain why).
- **`data/processed/aligned_raw.zip` rebuilt** from the corrected, cleaned 201-patient data (405 files: 1 CSV + 201 images + 201 masks + 2 dir entries) — ready for upload to Kaggle as a new dataset version.
- **`CLAUDE.md` updated:** new §1.4.4 (full writeup), §1.4.2/§1.4.3 counts marked superseded (kept for the record, not deleted, per this project's convention), §3.2b's "bimodal... 75% cluster" claim corrected with real post-fix statistics.
- **Not yet done:** the full `verify_alignment_sanity_check.ipynb`-style visual re-review hasn't been re-run against the regenerated data (only `Italy_022`/`Italy_004` were visually spot-checked, in `notebooks/find_corrupted_masks.ipynb`) — recommended before the next Kaggle run, especially across more of the 29 newly-fixed white-background patients.

## Where things stand
The aligned-photo segmentation dataset went through two major corrections this session and is now considered solid:

1. **v1 alignment (`cv2.matchTemplate`) was proven wrong.** A manual visual review of `India_071` found its mask floating on the sclera instead of the lower eyelid, despite a 0.995 confidence score — appearance correlation doesn't guarantee correct location. Discarded entirely.
2. **v2 alignment (SIFT/ORB + `cv2.findHomography(..., RANSAC)`)** replaced it — geometrically constrained, not a single appearance score. Original result: 202/217 aligned, 15 honest failures. Visually confirmed correct on `India_071` + several spot-checks. Full writeup: `CLAUDE.md` §1.4.1 (v1) / §1.4.2 (v2).
3. **The failures are permanently excluded**, not manually fixed. A `cv2.selectROI` manual-bbox tool was built and actually run, but the project author rejected the approach (crude/inconsistent vs. the RANSAC-derived masks) — tool and its output discarded. `AlignedConjunctivaSegmentationDataset` filters to `status == "ok"` without touching the shared `dataset_splits.csv` (`ConjunctivaSegmentationDataset`/`AnemiaClassificationDataset` still see all 217). Full detail: `CLAUDE.md` §1.4.3.
4. **Superseded by the white-background mask fix (§1.4.4, see the ✅ RESOLVED section above):** current numbers are **201/217 aligned, 16 failed** (13 India, 3 Italy), `aligned_seg_*` = 143/27/31 (201). The 202/15 numbers above are the pre-fix historical record, not the current state.

**Training is authorized to proceed on the aligned dataset** (now 201 patients post-fix, see the ✅ RESOLVED section above). The CLAUDE.md §1.4.2 gate ("no training until the project author has personally reviewed `verify_alignment_sanity_check.ipynb` and confirmed it") was explicitly cleared by the author on 2026-07-18 for the pre-fix 202-patient data — that review has not been formally repeated against the regenerated 201-patient data (only 2 of the 29 newly-fixed patients were spot-checked), though the fix itself is quantitatively and visually verified.

## The collapse mystery — two fixes applied, root cause still not 100% certain
An early Kaggle training attempt on the (since-discarded, v1) aligned dataset collapsed to all-background (`val_dice` pinned at `0.0000`, `val_loss` stuck ~0.60–0.61). Two things happened in response:
- **Fix 1 — `BCEDiceLoss`** (`CLAUDE.md` §3.2a): replaced plain `BCEWithLogitsLoss`, whose per-pixel averaging under-penalizes a rare foreground. Verified: penalizes an all-background collapse 25× harder than plain BCE did, on synthetic data matching the real sparsity.
- **Re-interpretation:** the collapsed run was against v1's *wrong* masks. A model trained on non-blank-but-spatially-wrong targets would plausibly collapse regardless of loss function (no learnable image→mask relationship). So it's genuinely unknown whether `BCEDiceLoss` alone would have fixed it, or whether the real problem was always the bad masks.
- **Fix 2 (this session) — `FocalTverskyLoss` + side-by-side comparison** (`CLAUDE.md` §3.2b): rather than assume `BCEDiceLoss` is sufficient, `loss_fn` is now itself an Optuna-tuned categorical hyperparameter (`bce_dice` vs. `focal_tversky`), so the next Kaggle run will directly compare both on the clean, current v2 data. `FocalTverskyLoss` (α=0.3, β=0.7, γ=4/3) targets the same failure mode more aggressively — verified to penalize an all-background collapse even harder than `BCEDiceLoss` (0.9999 vs. 0.0013, vs. `BCEDiceLoss`'s 0.510 vs. 0.0057) and to punish a 50%-recall partial miss substantially more too (0.311 vs. 0.177).
- **Correction:** the "bimodal, small Italy-crop cluster at 75%" sparsity claim previously here was measuring the white-background mask bug (see ✅ RESOLVED above), not real anatomy. Corrected, current stats on the 201-patient set: median 3.80% foreground, unimodal, range 1.78%–9.77% — no bimodal cluster. `FocalTverskyLoss`'s γ>1 down-weighting is still a reasonable general choice for sparse segmentation but its original specific bimodal justification no longer holds (not re-litigated).

**This has not been run on Kaggle yet.** Nothing is confirmed until real training data comes back.

## What's working right now
- Full Phase 0 pipeline, reproducible from `archive.zip` (`scripts/phase0_prepare_dataset.py`).
- `scripts/dataset.py`: three dataset classes, stratified split builder, transforms, `get_dataloaders()`. `AlignedConjunctivaSegmentationDataset` correctly filters to the 201 aligned patients (no more `FileNotFoundError` gap).
- `scripts/build_aligned_dataset.py` (v2, SIFT/ORB + homography, now with the white-background mask fallback — CLAUDE.md §1.4.4): 201/217 aligned, 16 honest failures, inliers 5–2014 (mean 312.5). Also now auto-cleans orphaned output files from patients that flip from ok→failed across runs.
- Three segmentation architectures (`models/segmentation/{unet,attention_unet,resunet}.py`), forward-pass verified on GPU.
- `scripts/trainer_engine.py`: model-, dataset-, *and now loss-function*-agnostic Optuna engine. `loss_fn` sampled per-trial from `LOSS_REGISTRY` (`bce_dice`, `focal_tversky`); each gets its own checkpoint (`best_{model_name}_{loss_fn}.pth`) plus a per-loss comparison table in the study summary JSON. Verified via a synthetic 6-trial run with mocked results.
- Six entry-point scripts (3 original crop-based + 3 `_aligned`) — no changes needed for the loss comparison, since loss selection lives entirely inside the shared engine now.

## Training results that exist right now
**Original crop-based dataset, all 3 models trained on Kaggle T4×2 (before the loss-fn-as-hyperparameter change, under plain `BCEDiceLoss`):**
- Real per-trial logs: `outputs/logs/{unet,attention_unet,resunet}_trials.csv` + `_study_summary.json`.
- Standard U-Net: a Kaggle result (Trial 4, Val Dice 0.9900, Val IoU 0.9800) was reported but **not independently verified** — no notebook/log artifact available. A local partial run showed different numbers (Trial 0 best, Dice 0.9893). Both recorded in `CLAUDE.md` §3.5/§3.6 with explicit provenance labels — don't cite the 0.99/0.98 figures as confirmed.

**Aligned dataset (v2, 201 patients post-fix, with the loss comparison):** no training has been run yet. (An earlier Kaggle attempt against the pre-fix 202-patient data, contaminated by the white-background mask bug, was manually stopped by the project author before completion — no results from it should be used.)

**Other settled infrastructure decisions (still current):** `n_trials=12` for the three `_aligned` entry-point scripts (was engine default of 5) — see `CLAUDE.md` §3.4. No Kaggle CLI in this environment; upload and running the 3 `_aligned` scripts remains a manual, external step for the project author. `my_valuable_outputs.zip` (351MB, untracked) is a harmless backup of the original crop-based Kaggle run's checkpoints/logs (§3.6).

## Immediate next step
1. **Project author uploads the freshly-rebuilt `data/processed/aligned_raw.zip` to Kaggle** (new dataset version — contains the corrected 201-patient data, mask bug fixed). `git pull` the latest code (includes the mask fix + `n_trials=12`).
2. Run the 3 `_aligned` entry-point scripts on Kaggle (12 trials each, per-loss-function comparison built in).
3. Pull `outputs/checkpoints/best_*_aligned*.pth` + `outputs/logs/*_aligned_*` back into this repo.
4. Compare `bce_dice` vs. `focal_tversky` via the per-loss comparison table in the study summary JSON.
5. **Re-run a domain-shift check** — does the winning model actually isolate tissue on a raw photo? This is still the entire point of the original pivot; don't skip it.
6. Only after that: rebuild Phase 3 (real inference + cropping) against whichever model generalizes.
7. Phase 4 (classification) hasn't been started at all yet.
8. Optional but recommended before the Kaggle run: re-run the `verify_alignment_sanity_check.ipynb`-style visual review against the regenerated `aligned_raw/` (only 2 of the 29 newly-fixed patients have been visually spot-checked so far).
