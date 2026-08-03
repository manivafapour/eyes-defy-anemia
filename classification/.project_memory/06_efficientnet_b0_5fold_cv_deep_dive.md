# EfficientNet-B0 (forniceal_palpebral) — 5-Fold CV Deep Dive

**Directory:** `classification/datapreparepipeline/efficientnet_b0_forniceal_5fold_cv/` (originally built at `classification/scripts/efficientnet_b0_forniceal_5fold_cv/`, moved when `scripts/` was renamed to `datapreparepipeline/` on 2026-07-31 — same content, new path).

**Consolidation note:** this file pulls together everything about this specific piece of work — previously scattered across five separate dated entries in `02_current_status.md` (2026-07-30 through 2026-08-01) plus one line in `01_roadmap.md` — into one standalone, thesis-readable account. Nothing here is new information beyond what was already recorded in this project's memory or directly verified via tool calls earlier in the same working session; it is reorganized and, in the Grad-CAM section, filled out with quantitative detail that existed in the working session but had only been summarized tersely in `01_roadmap.md`. The original dated entries in `02_current_status.md` are left untouched as the historical record.

**Timeline:** started 2026-07-30, real training run completed and analyzed 2026-08-01.

---

## 1. Why this happened

After Batch 1 of the 18-combo v2 sweep completed (2026-07-29), the project author paused the remaining batch-2 work to do a focused optimization/validation pass on the single best-performing combo: **EfficientNet-B0 / forniceal_palpebral**, which had scored val F1 = 0.933 in Batch 1 — the highest of all 14 Batch-1 combos. The goal was a proper 5-fold cross-validation of that specific (architecture, tissue_type) pair, rather than trusting a single Optuna train/val split's number. The full roadmap (warm-start vs. fresh-init, hyperparameter source, augmentation baseline, test-set handling, Optuna scope) was discussed and confirmed before any code was written.

## 2. Pipeline design (`cv_dataset.py`, `cv_trainer_engine.py`, `run_cv_training.py`)

Three new files, named with a `cv_` prefix specifically to avoid a Python module-name collision with the shared `dataset.py`/`trainer_engine.py` they import from (two files sharing a bare module name on `sys.path` at once would silently resolve to whichever loaded first).

**Confirmed design decisions:**
1. **Fresh, randomly-initialized head per fold — no warm-start** from the Batch-1 checkpoint. This resolves an initialization-leakage concern: that checkpoint's weights were selected via its own validation F1 on the *original* split, which could partially overlap with any given fold's training data. `build_efficientnet_b0()` (reused from the shared engine) already constructs a fresh random `Dropout → Linear` head on every call.
2. **Hyperparameters locked from Batch 1's winning trial (#9, val F1 = 0.933), Optuna dropped entirely:** `learning_rate = 0.0008200518402245837`, `weight_decay = 1.9634341572933354e-06`, `dropout_rate = 0.2`.
3. **Root cause of Batch 1's jagged loss curves identified empirically, not assumed:** per-trial validation-loss volatility correlated against both hyperparameters across all 12 real Batch-1 trials. Learning rate correlated at 0.998 (near-perfect); dropout correlated at −0.09 (essentially none). Locking LR at trial #9's value (one of the lowest-volatility trials) directly addresses the dominant cause.
4. **Augmentation baseline unchanged**, explicitly re-verified against the live `dataset.py` code rather than assumed from memory: `HorizontalFlip(p=0.5)` + `Rotate(limit=15°, border_mode=0, fill=0, p=0.5)` on train only; `Normalize` (ImageNet stats) + resize on both train and eval. No color/brightness jitter — a deliberate choice, independently reinforced by the Ramos-Soto et al. literature precedent (`04_literature_review_findings.md`).
5. **Test set (33 patients) permanently excluded**, not filtered per fold — removed once in `load_cv_pool()` before fold construction, so no code path in this pipeline can ever see them. **CV pool is 178 patients**, corrected from an initial miscount of 184 (184 was train+val before filtering out the 6 patients dataset-wide who lack a forniceal_palpebral crop at all). Smallest stratum (Italy, anemic) is 19 patients — checked as comfortably sufficient for `StratifiedKFold(n_splits=5)` before writing the splitting code.
6. **New training-loop mechanics**, explicitly flagged at the time as the assistant's own chosen defaults rather than independently specified in the roadmap: gradient clipping (`max_norm=1.0`); `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)`; `EARLY_STOPPING_PATIENCE=15` (deliberately more than double the scheduler's patience, so early stopping cannot fire before the scheduler gets a chance to act first). Batch size 32 (up from the shared engine's 16 at the time), 150-epoch ceiling.

**Verified before considering the pipeline done:** both new modules import cleanly with no naming collision (checked directly). A real 2-fold/2-epoch dry run (`MAX_EPOCHS` monkey-patched, not a code change) ran the complete loop end-to-end — gradient clipping, `ReduceLROnPlateau.step()`, checkpoint saving, all 5 plot types per fold, per-fold JSON history, cross-fold aggregation — with zero errors. Dry-run artifacts deleted afterward, per this project's standard convention.

## 3. Kaggle notebook (`efficientnet_b0_5fold_cv.ipynb`)

Confirmed to run on Kaggle (standard practice for real compute in this project, same as every other batch).

`run_cv_training.py` was extended with `--fold N` (run exactly one fold, no aggregation) and `--aggregate` (compute the cross-fold summary from already-saved per-fold history JSONs, no training) CLI modes, specifically to restore the same per-cell `sync_outputs()` incremental-safety pattern the earlier Batch 1/2 notebooks had — the original single-process, all-5-folds design had no protection against a mid-run Kaggle interruption, a real risk given this is by far the longest run in the project (5 folds × up to 150 epochs). `aggregate_only()` was verified to fail loudly (a `FileNotFoundError` listing exactly which folds are missing), not silently, if any fold's history is absent — tested directly with 0/5 and 2/5 folds present.

The notebook itself is 20 cells: Setup/Data reused verbatim from the already-Kaggle-verified `classification-final-fixed.ipynb`, a CV-specific sanity check (`load_cv_pool()` + `build_folds()`, asserts the pool is exactly 178 patients), 5 per-fold training cells + 1 aggregate cell each followed by `sync_outputs()`, zipping to `efficientnet_b0_cv_results.zip`.

**Verified before calling it done:** `nbformat.validate()` passed. A real incremental dry run (`MAX_EPOCHS` monkey-patched to 2) exercised the exact sequence the real notebook would run — folds 1 and 3 first (deliberately out of order and partial), confirmed `aggregate_only()` correctly refused to run with folds missing, then completed folds 2/4/5 and aggregated for real. All 5 folds produced the correct, differently-sized stratified splits: 142/36, 142/36, 142/36, 143/35, 143/35 (sums to 178) — and all 5 plot types each. Dry-run artifacts deleted afterward.

## 4. The real run — headline results

Executed on Kaggle by the project author; results placed in `outputs/{checkpoints,logs,plots}/`. **All 5 folds early-stopped between epoch 37 and 69 — none reached the 150-epoch ceiling.** Every number below was computed directly from the 5 `*_fold{N}_history.json` files and `*_cv_summary.json`, not estimated. Full report: `outputs/CV_5FOLD_ANALYSIS_REPORT.md` (also in this same directory).

**Mean ± SD across the 5 folds:**

| Metric | Value |
|---|---|
| Overall F1 | 0.867 ± 0.053 |
| Overall AUC | 0.876 ± 0.066 |
| Overall Balanced Accuracy | 0.884 ± 0.053 |
| **India AUC** | **0.639 ± 0.057** (range 0.564–0.729) |
| Italy AUC | 0.922 ± 0.098 (high variance — fold 4 alone drops to 0.767) |

**Every one of the 5 folds underperforms the single-split Optuna result this run was meant to validate** (India AUC 0.750, from the Batch-1 trial #9 study summary). This is exactly the failure mode 5-fold CV exists to catch. **Conclusion recorded at the time: the CV mean (0.639 ± 0.057), not the single-split number (0.750), is the figure to cite for India-cohort claims going forward.**

**Latency** (not tracked by the training pipeline itself — measured locally for the report, on the project's RTX 4050): batch=1 GPU 13.9ms mean / CPU 26.6ms mean; batch=32 throughput ≈663 images/sec (1.5ms/image amortized). Identical across all 5 folds by construction (same architecture, only learned weights differ) — not a deployment concern at this latency.

## 5. Overfitting analysis: Folds 1 and 4 vs. Fold 2 (healthy baseline)

Both folds the project author had flagged from the plots were confirmed as real overfitting, not a false alarm — verified by exactly replaying the early-stopping/`ReduceLROnPlateau` state machine against each fold's logged `val_loss_history`/`lr_history` arrays; the reconstructed stop epoch matched the logged `n_epochs_run` in all 5 folds, cross-validating the whole analysis.

- **Fold 4 — sharp overfit.** Last real validation-loss improvement at epoch 22 (val loss 0.605 — also the *worst* best-case val loss of any fold by a wide margin). Final train/val loss gap +0.145 (the largest of all 5 folds). Three learning-rate halvings occurred after that point, none of which recovered generalization.
- **Fold 1 — milder, slower version of the same pattern.** Last improvement at epoch 23, final gap +0.071 (about half of Fold 4's).
- **Folds 2, 3, 5 — no divergence at all.** Validation loss tracks at or below training loss for the entire run, consistent with the expected asymmetry between augmented training data and clean validation data (not data leakage).
- **Root cause: per-fold train/val split composition** (ordinary small-N k-fold variance on a 178-patient pool) — **not the fixed hyperparameters**, since the identical learning rate/weight decay/dropout produces clean convergence on 3 of the 5 folds.

**Nuance on the India/Italy AUC gap specifically:** Fold 4 has the worst India AUC of any fold, but *not* the worst India/Italy gap (0.203) — Italy AUC also collapses in that same fold (0.767), so Fold 4 is a generally hard split for both cohorts, not an India-specific amplifier. The starkest *relative* disparities are actually Folds 3 and 5, where Italy AUC is a perfect 1.000 while India sits at 0.600/0.636 — cases where the aggregate metric (F1 ≈0.88, AUC ≈0.90–0.91) looks strong and would mask the India-specific shortfall if the country-stratified breakdown weren't checked.

## 6. The core finding: India-cohort shortcut-learning evidence

**India recall/sensitivity is exactly 1.000 in all 5 independent folds — zero false negatives across all 57 India-anemic validation patients, across 5 differently-initialized models trained on 5 different splits.** Combined with India specificity of only 0.35 ± 0.11, and a near-constant per-fold `pos_weight` (1.31–1.37, which rules out fold-specific hyperparameter noise as the explanation), this is strong, reproducible evidence of a near-deterministic "predict anemic for India" pattern rather than genuine per-patient discrimination. It is the clearest version yet of the country-correlated shortcut pattern already documented across the original 6-combo and 18-combo v2 comparisons — now shown, for the first time, to survive 5 independent re-trainings rather than being an artifact of one lucky/unlucky split.

## 7. Grad-CAM diagnostic (`gradcam_analysis.ipynb`)

Built and run locally (not on Kaggle) specifically to test whether the India shortcut is a genuine visual shortcut (attention on non-tissue cues) or more of a calibration/base-rate artifact, using the Fold 4 checkpoint (the fold with the sharpest overfitting and the worst India AUC).

**Correctness check:** the notebook reconstructed Fold 4's validation set from scratch (same `StratifiedKFold(seed=42)` used in training) and reproduced its exact confusion matrix (India TN=2, FP=3, FN=0, TP=11) before trusting anything downstream — confirming the reconstruction matched what was actually trained, not an approximation.

**Tissue-attention-ratio quantification** (fraction of each Grad-CAM heatmap's activation energy falling inside a segmented tissue region vs. background), computed across three groups — all 3 India false positives, all 11 India true positives, and all 14 correctly-classified Italy patients in Fold 4's validation set:

| Group | n | Mean | Std |
|---|---|---|---|
| India False Positive | 3 | 0.154 | 0.106 |
| India True Positive | 11 | 0.258 | 0.104 |
| Italy Baseline | 14 | 0.171 | 0.208 (high — see below) |

The Italy Baseline group's high variance turned out to be a real bimodal split, not noise, once cross-referenced against the white-background flag (see §8 below): Italy patients using the white-background convention (n=3) had a tight, moderate mean of 0.257; Italy patients using the standard black-background convention (n=11) were sharply bimodal — **7 of those 11 (64%) had a tissue-attention ratio below 0.02**, meaning the model's attention sat almost entirely on background padding, not tissue, while still landing on the correct prediction. **Zero of the 14 India patients (false positive or true positive) showed this near-zero pattern** — India's worst case was still 0.032.

**Interpretation recorded at the time:** this reframes the finding beyond a simple "India has a shortcut, Italy behaves normally" story. The "normal behavior" baseline (Italy) is itself inconsistent — much of the time it gets the right answer while barely attending to tissue at all, plausibly by leaning on Italy's low base rate (81% non-anemic dataset-wide) as a safe default rather than genuine pallor discrimination. India cannot fall back on that same trick, since its base rate points the opposite way (72% anemic) — the same underlying failure to robustly read tissue, expressed as opposite surface symptoms depending on which country's prior a given patient happens to share.

## 8. Data bug discovered as a direct byproduct of this work

While building the Grad-CAM tissue mask (which needed to correctly distinguish tissue from background to compute the ratio above), a naive black-background threshold was found to silently mis-measure a subset of the Italy baseline group. Investigating why led directly to discovering that **`prepare_dataset.py`'s `flatten_to_black()` had a real bug**: a subset of source crops (30/211 forniceal_palpebral patients — 25.9% of Italy, 0% of India) use an opaque white background instead of alpha transparency, the same bug class the segmentation phase had already found and fixed months earlier in its own, independent pipeline. This CV/Grad-CAM run's results were trained on the **pre-fix** images.

This is documented in full detail (root cause, the fix, and its independent re-verification — 428/428 images passing 4 strict checks) in `02_current_status.md`'s "Data bug fixed" entry and is the direct motivation for the clean-data 18-combo retrain now tracked in `05_kaggle_training_phase.md`. Not repeated in full here to avoid duplicating that record — this section exists only to make the causal link explicit: **the bug was found *because of* the diagnostic work in this directory, not independently of it.**

## 9. Mitigation strategies identified (none implemented in this section's scope)

Ranked by recommended investigation order at the time (full rationale in `outputs/CV_5FOLD_ANALYSIS_REPORT.md`):

0. **Diagnose first:** Grad-CAM/saliency inspection on India false positives (done, above) before investing in any fix.
1. **Data-centric:** country-balanced mini-batch sampling; HSV+CLAHE illumination normalization (literature-backed, `04_literature_review_findings.md`); more India data (the smaller, harder cohort).
2. **Algorithmic:** domain-adversarial training (DANN-style — the most direct answer to "unlearn the shortcut"); Group-DRO/worst-group loss; within-dataset CORAL (already flagged as untested in the literature review); partial backbone fine-tuning with stronger regularization.
3. **Cheap/stopgap, explicitly not a real fix:** per-country post-hoc threshold recalibration — flagged at the time as unable to move AUC at all, since AUC is threshold-independent; a deployment-time patch, not a discrimination fix.

A separate theoretical consultation (outside this directory's own file scope, not repeated in full here) evaluated combining HSV+CLAHE with DANN specifically, and raised a load-bearing caveat worth remembering for later: **every backbone in this project's pipelines is frozen** — only a small linear head trains — so the classic "ViT/DANN fine-tuning instability" concern from the general literature does not transfer cleanly to this setup, and DANN in particular would need the frozen-backbone decision revisited to make architectural sense at all.

**Not yet done, as of this file's writing:** deciding which mitigation strategy (if any) to pursue; whether to retrain this specific combo (EfficientNet-B0/forniceal_palpebral, 5-fold CV) against the now-fixed clean data to see whether/how much the bug contributed to the India AUC problem, separately from the broader 18-combo clean-data retrain already underway.

## 10. Where everything lives

- Pipeline code: `classification/datapreparepipeline/efficientnet_b0_forniceal_5fold_cv/{cv_dataset.py, cv_trainer_engine.py, run_cv_training.py}`
- Kaggle notebook: `.../efficientnet_b0_5fold_cv.ipynb`
- Real run outputs: `.../outputs/{checkpoints,logs,plots}/` (5 checkpoints, per-fold history JSONs + `cv_summary.json`, all plots)
- Full standalone analysis report: `.../outputs/CV_5FOLD_ANALYSIS_REPORT.md`
- Grad-CAM notebook + its outputs: `.../gradcam_analysis.ipynb`, `.../outputs/gradcam/` (heatmap PNGs, `fold4_val_predictions.csv`, `fold4_tissue_attention_ratios.csv`, boxplot)
