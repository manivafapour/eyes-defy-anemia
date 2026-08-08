# Kaggle Training Phase — Clean-Data 18-Combo Sweep (classification/, Phase 4)

**Started:** 2026-08-02. First file under the new one-file-per-phase documentation convention (`03_tech_stack_and_rules.md` rule #9) — everything from here on about *this specific training phase* goes in this file, not appended to `02_current_status.md`.

## What this phase is

Retraining the full 18-combo v2 architecture sweep (9 architectures × 2 tissue types) against the dataset after the white-background bug fix, to establish bug-free baselines and find out whether/how much that bug was contributing to the India/Italy AUC gap and shortcut-learning behavior investigated over the prior sessions.

**Context this phase builds on** (full detail in `02_current_status.md`'s "READY FOR TRAINING" entry and earlier dated entries — not repeated here):
- White-background convention bug found and fixed in `prepare_dataset.py`; full dataset reprocessed; independently re-verified clean (428/428 images, 4 strict checks, zero failures).
- WHO labeling thresholds re-examined against Ramos-Soto et al. and reconfirmed (adopting their thresholds would widen the India/Italy gap, not narrow it).
- 18 new entry-point scripts (`classification/v2_clean_scripts/train_*_v2_clean.py`) and 2 Kaggle notebooks (`classification-cnn-clean.ipynb`, `classification-vit-clean.ipynb`) built, committed, and pushed.

## Status

**Both batches complete — full 18-combo clean-data sweep done (2026-08-04).** `classification-cnn-clean.ipynb` (12 CNN combos) and `classification-vit-clean.ipynb` (6 transformer combos) have both been run on Kaggle, downloaded, organized, and merged into one comparison. This maps onto the project author's stated 3-phase roadmap (Phase 1: this 18-combo Optuna sweep → Phase 2: select top-2 CNN + top-2 transformer → Phase 3: 3-fold CV on the 4 champions) as: **Phase 1 complete, Phase 2 unblocked.**

## Naming reference (for when results start coming in)

All 18 new results use the `_v2_clean` model-name suffix (e.g. `resnet18_palpebral_v2_clean`), distinct from both the original v1 and the dirty-data v2 results — all three generations stay on disk side by side, nothing gets silently overwritten.

## Log

### 2026-08-02 — Pre-launch config tuning: `n_startup_trials` and `batch_size`

Two changes to the shared engine before launch, both in code every one of the 18 `v2_clean_scripts` entry points inherits automatically (they call `run_study()` with no hyperparameter arguments of their own — all config lives in `trainer_engine.py`/`dataset.py`, not duplicated per script):

1. **`trainer_engine.py`: `TPESampler(seed=SEED, n_startup_trials=5)`**, was the Optuna default of 10. With `N_TRIALS=12`, the default meant 10/12 trials per combo were pure random sampling before TPE's Bayesian modeling ever engaged — verified directly against the installed Optuna version's actual default (`inspect.signature`, not assumed from general knowledge). 5 leaves a real 7-trial informed-search budget instead of 2. Applies identically to all 18 combos, not architecture-family-specific — a considered CNN-vs-Transformer LR-bounds split was discussed and deliberately deferred (see rationale below).
2. **`dataset.py`: `BATCH_SIZE` raised 16 → 32.** Safe because every backbone is frozen (only the head trains, 441–1,281 trainable params depending on architecture) — comfortable VRAM headroom on Kaggle's GPUs, halves optimizer steps/epoch across the sweep.

**Considered and deliberately NOT done: splitting the Optuna LR search bounds by architecture family** (narrower/lower range for the 3 transformers). Reasoning: the classic "ViT fine-tuning is unstable at high LR" risk is about training self-attention weights directly, which doesn't apply here since the backbone is frozen for every architecture — what's actually being tuned is a small linear head on frozen features, a much more benign optimization landscape. Splitting the bounds now, without pilot evidence, would also risk making Phase 1's architecture comparison unfair (Phase 2 selects top-2 CNN + top-2 transformer from these results) — a search-design asymmetry could get conflated with genuine architecture quality. Deferred to a possible evidence-based, narrower re-search scoped to just the Phase 2 champions before Phase 3's 3-fold CV, if Phase 1's own per-trial logs (`learning_rate` vs. `val_f1`) show real divergence patterns for transformers once results are in.

Verified before considering this done: both files `py_compile` clean; runtime check confirms `dataset.BATCH_SIZE` (32) propagates through `trainer_engine.py`'s import; a `v2_clean_scripts` entry point (`train_resnet18_palpebral_v2_clean.py`) confirmed to see `BATCH_SIZE=32` via the shared module, not a stale copy.

### 2026-08-02 — CNN batch complete: results organized, compared, and committed

Project author confirmed `classification-cnn-clean.ipynb` finished running on Kaggle, downloaded the results zip, and placed/unzipped it at `classification/v2_clean_scripts/outputs/` (flat `checkpoints/`/`logs/`/`plots/`, matching the notebook's `sync_outputs()` layout — verified: 12 checkpoints, 24 logs, 48 plots, all 12 `study_summary.json` files parse).

**`organize_and_compare.py` written** (`classification/v2_clean_scripts/`) — discovers combos dynamically from `outputs/logs/*_study_summary.json` filenames rather than a hardcoded architecture list, so the same script works unchanged for the ViT batch later. Per combo: moves checkpoint + 2 logs + 4 plots into `outputs/{combo_name}/` (7 files each), writes `metrics_summary.json`+`.md` (F1/Accuracy/Recall/Precision/Specificity/Balanced Accuracy as requested, plus AUC and the India/Italy breakdown as bonus context), and builds `outputs/model_comparison/comparison_table.csv`+`comparison_report.md` ranked by F1. Run for real (not just written) — output verified against the actual files afterward.

**CNN batch results (clean data, all 12 combos, sorted by F1):**

> **Caveat added 2026-08-04 — read before citing the `India/Italy AUC Gap` column.** Every India AUC behind this column is computed on 40 discordant pairs (10 anemic × 4 healthy in the 33-patient val split), giving a 95% CI half-width of roughly ±0.27 — wide enough to swallow the entire observed spread. The per-combo gap ordering below is therefore **not** a reliable ranking of confound handling, and the gap range 0.0192–0.4500 should not be read as 12 distinguishable results. What *is* solid is the systematic direction (India AUC < Italy AUC in 11 of 12 models, sign test p=0.0064) and the tissue-type effect (palpebral beats forniceal_palpebral on India AUC in 5 of 6 paired architecture comparisons, mean +0.121). The F1/Balanced-Accuracy/AUC-overall columns are computed on the full 33 patients and are less affected, though still single-split point estimates. `07_step1_measurement_harness.md` replaces this estimator with a 1,311-pair pooled out-of-fold one; re-derive any confound claim against that once it has run.

| Rank | Model | F1 | Balanced Acc. | AUC | India/Italy AUC Gap |
|---|---|---|---|---|---|
| 1 | ConvNeXt-Tiny / palpebral | 0.9333 | 0.9474 | 0.9398 | 0.1000 |
| 2 | EfficientNet-B0 / forniceal_palpebral | 0.9032 | 0.9118 | 0.8824 | 0.3365 |
| 3 | RegNetY-400MF / forniceal_palpebral | 0.8966 | 0.9055 | 0.8739 | 0.4500 |
| 4 | RegNetY-400MF / palpebral | 0.8750 | 0.8947 | 0.9173 | 0.2417 |
| 5 | EfficientNet-B0 / palpebral | 0.8667 | 0.8853 | 0.9323 | 0.2167 |
| 6 | DenseNet121 / forniceal_palpebral | 0.8485 | 0.8529 | 0.8782 | 0.3058 |
| 7 | ResNet18 / palpebral | 0.8387 | 0.8590 | 0.8910 | 0.2167 |
| 8 | DenseNet121 / palpebral | 0.8387 | 0.8590 | 0.8872 | 0.3583 |
| 9 | ConvNeXt-Tiny / forniceal_palpebral | 0.7778 | 0.7647 | 0.7437 | 0.2904 |
| 10 | MobileNetV3-Small / palpebral | 0.7742 | 0.7970 | 0.8759 | 0.1083 |
| 11 | ResNet18 / forniceal_palpebral | 0.7692 | 0.7983 | 0.7731 | 0.2750 |
| 12 | MobileNetV3-Small / forniceal_palpebral | 0.7568 | 0.7353 | 0.7941 | **0.0192** |

**Top performer: ConvNeXt-Tiny/palpebral** (F1=0.9333, Balanced Accuracy=0.9474, AUC=0.9398) — also a genuinely good India/Italy AUC gap (0.100), not just a good headline number. **Best confound-handling: MobileNetV3-Small/forniceal_palpebral** (gap=0.0192, smallest of all 12) but much weaker F1 (0.7568) — same "best-overall vs. best-confound-handling disagree" pattern as every prior sweep in this project (original 6-combo, dirty-data 18-combo v2). Not yet compared against the *dirty-data* v2 results for these same 12 (architecture, tissue) pairs to see whether/how much the white-background fix changed rankings — that comparison is a natural next step once useful, not yet done.

**Two more gitignore anchoring gaps found and fixed** (3rd and 4th instances of the same issue this project keeps hitting — `outputs/*/*.pth` is anchored to `classification/`'s own directory level and does not reach deeper nested locations):
- `v2_clean_scripts/outputs/*/*.pth` — found while organizing the CNN batch results above, before any checkpoint could be staged.
- `v2_scripts/outputs/*/*.pth` — found when the project author manually relocated `classification/outputs/` (the *original dirty-data* 18-combo v2 results + `v2_comparison_results/`) to `classification/v2_scripts/outputs/`, to sit alongside the scripts that produced it (mirroring the `v2_clean_scripts/` layout). This one mattered more: 18 checkpoint files, some very large (e.g. ViT-L/16 ≈1.2GB), were briefly stageable before the fix.

**That manual relocation verified content-lossless before staging, not assumed:** hash-compared a JSON and a PNG (byte-identical), and traced the one file that showed a hash difference (`v2_comparison_results/comparison_table.csv`) down to a pure CRLF/LF line-ending artifact (19 lines, 19 extra bytes, zero semantic change — confirmed via `od -c` byte inspection) rather than a real edit. Git detected all 114 moved files as clean 100% renames.

**Committed (`c6f03a0`):** the relocation (114 renames), the gitignore fixes, and the organized CNN batch results (`organize_and_compare.py` + `v2_clean_scripts/outputs/`, 12 combos + `model_comparison/`) — 214 files, verified zero files over 5MB before committing.

**A second Kaggle-pushed commit found via the now-standard fetch-before-push check** — the *fully executed* `classification-cnn-clean.ipynb` (9,020 insertions, real outputs in 19 code cells). Verified cell sources were byte-identical to local before merging (zero divergence to reconcile) — purely additive, merged cleanly (`520e29f`), `nbformat.validate()` passed. Pushed; local and remote confirmed in sync.

**Not yet done (as of the CNN-only milestone above):** ViT batch not yet run. Comparison against the dirty-data v2 results for the same 12 combos not yet done. Phase 2 (select top-2 CNN + top-2 transformer) waits on the ViT batch completing so all 18 combos can be compared together, same "analyze once, not twice" pattern already used for batch 1/batch 2 of the original v2 sweep.

### 2026-08-04 — ViT/Transformer batch complete: organized, merged into a full 18-combo comparison

Project author confirmed `classification-vit-clean.ipynb` finished on Kaggle and placed `vit_clean_results.zip` (3.10 GB, 45 entries — same flat `checkpoints/`/`logs/`/`plots/` layout as the CNN batch) at `classification/v2_clean_scripts/outputs/`. Extracted via `System.IO.Compression.ZipFile.ExtractToDirectory` (worked around a PowerShell cmdlet incompatibility with the boolean overwrite argument by omitting it, since the target subfolders didn't exist yet — no overwrite needed). Confirmed 6 combos present before organizing: `swin_t`/`vit_b_16`/`vit_l_16` × `palpebral`/`forniceal_palpebral`.

**`organize_and_compare.py`'s dynamic combo-discovery worked exactly as designed** (`02_current_status.md`'s design note: "discovers combo names from `outputs/logs/*_study_summary.json` — not a hardcoded architecture list — so this same script works unchanged for the ViT batch later") — ran `organize_outputs()` unmodified, it found only the 6 newly-extracted combos (the 12 CNN combos were already moved out of the flat `logs/` dir from the earlier run, so they were correctly left untouched) and moved each into its own `outputs/{combo_name}/` folder (checkpoint + 2 logs + 4 plots + computed `metrics_summary.json/.md`, 9 files each, matching the CNN batch's layout exactly).

**Comparison rebuilt to cover all 18 combos, not just the 6 new ones** — `build_comparison()` only receives whatever combo list it's called with, so a plain re-run of `organize_and_compare.py`'s `main()` would have silently overwritten `model_comparison/` with a 6-row table and dropped the 12 CNN results. Instead, scanned `outputs/` for every subdirectory (excluding `model_comparison/`) containing its own `*_study_summary.json`, extracted metrics for all 18, and rebuilt `comparison_table.csv` + `comparison_report.md` from the full set. Also fixed the report's title, which was hardcoded as "CNN Batch -- Model Comparison" from when only CNN results existed — renamed to "Full 18-Combo Model Comparison" (`organize_and_compare.py` line ~181) since it's no longer CNN-only, and the script will keep producing an accurate title for any future re-run.

**One incidental blocker, resolved:** `comparison_table.csv` was open in Excel (project author had it open for inspection), which raised a `PermissionError` on the first write attempt — write succeeded once the file was closed, no data lost.

**Full 18-combo results (sorted by F1):**

| Rank | Model | F1 | Balanced Acc. | AUC | India/Italy AUC Gap |
|---|---|---|---|---|---|
| 1 | ConvNeXt-Tiny / palpebral | 0.9333 | 0.9474 | 0.9398 | 0.1000 |
| 2 | ViT-B/16 / palpebral | 0.9333 | 0.9474 | 0.9098 | 0.1333 |
| 3 | EfficientNet-B0 / forniceal_palpebral | 0.9032 | 0.9118 | 0.8824 | 0.3365 |
| 4 | ViT-L/16 / palpebral | 0.8966 | 0.9117 | 0.9173 | 0.0583 |
| 5 | RegNetY-400MF / forniceal_palpebral | 0.8966 | 0.9055 | 0.8739 | 0.4500 |
| 6 | RegNetY-400MF / palpebral | 0.8750 | 0.8947 | 0.9173 | 0.2417 |
| 7 | EfficientNet-B0 / palpebral | 0.8667 | 0.8853 | 0.9323 | 0.2167 |
| 8 | DenseNet121 / forniceal_palpebral | 0.8485 | 0.8529 | 0.8782 | 0.3058 |
| 9 | Swin-Tiny / palpebral | 0.8485 | 0.8684 | 0.8910 | 0.0500 |
| 10 | DenseNet121 / palpebral | 0.8387 | 0.8590 | 0.8872 | 0.3583 |
| 11 | ResNet18 / palpebral | 0.8387 | 0.8590 | 0.8910 | 0.2167 |
| 12 | ViT-B/16 / forniceal_palpebral | 0.8333 | 0.8571 | 0.8950 | 0.2000 |
| 13 | ViT-L/16 / forniceal_palpebral | 0.8276 | 0.8403 | 0.8109 | 0.2231 |
| 14 | Swin-Tiny / forniceal_palpebral | 0.8276 | 0.8403 | 0.8193 | 0.3115 |
| 15 | ConvNeXt-Tiny / forniceal_palpebral | 0.7778 | 0.7647 | 0.7437 | 0.2904 |
| 16 | MobileNetV3-Small / palpebral | 0.7742 | 0.7970 | 0.8759 | 0.1083 |
| 17 | ResNet18 / forniceal_palpebral | 0.7692 | 0.7983 | 0.7731 | 0.2750 |
| 18 | MobileNetV3-Small / forniceal_palpebral | 0.7568 | 0.7353 | 0.7941 | 0.0192 |

**Top performer (tied): ConvNeXt-Tiny/palpebral and ViT-B/16/palpebral** — identical F1 (0.9333) and Balanced Accuracy (0.9474); ConvNeXt-Tiny leads on AUC (0.9398 vs. 0.9098) and has the smaller confound gap (0.1000 vs. 0.1333), so it remains the single best headline result. **Best confound-handling unchanged: MobileNetV3-Small/forniceal_palpebral** (gap=0.0192), though notably **ViT-L/16/palpebral (0.0583) and Swin-Tiny/palpebral (0.0500)** now occupy the next-best confound-handling positions of any combo in the full 18 — the three best-confound combos besides MobileNetV3-Small are now all transformers on the palpebral crop, worth flagging as a candidate pattern (small n, not statistically tested) for Phase 2 selection.

**Committed:** not yet — pending a separate explicit go-ahead, same standing rule as always.

*(Further entries appended here as the training phase progresses — the dirty-vs-clean v2 comparison if performed, Phase 2 selection. Once this phase is complete and a new one starts, e.g. Phase 3's 3-fold CV on the champions, that gets its own `07_...` file in turn. Note: `06_efficientnet_b0_5fold_cv_deep_dive.md`, a separate file, consolidates earlier work that chronologically *preceded* this phase — sequential numbering here reflects file-creation order, not chronology.)*
