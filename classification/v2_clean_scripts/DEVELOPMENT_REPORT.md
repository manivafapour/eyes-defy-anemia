# Development Report: `classification/v2_clean_scripts/`

**Project:** EYES-DEFY-ANEMIA — Phase 4 (Anemia Classification from Eye Images)
**Scope of this report:** the design, construction, and execution of the clean-data 18-combo training pipeline at `classification/v2_clean_scripts/`, and the results it produced.
**Prepared:** 2026-08-04, grounded directly in `classification/.project_memory/` (files `01_roadmap.md`, `02_current_status.md`, `05_kaggle_training_phase.md`) and the executed outputs at `classification/v2_clean_scripts/outputs/`.

---

## 1. Overall Development Process

### 1.1 Initial goal

Phase 4 of the EYES-DEFY-ANEMIA project adds a second downstream task alongside conjunctiva segmentation: binary anemia classification directly from eye photographs, using the same 217-patient India/Italy cohort and WHO-threshold labels (Male Hgb < 13.0 g/dL, Female < 12.0 g/dL) established elsewhere in the project. `classification/` was built as a **fully isolated module** — its own data extraction, its own dataset/label logic, no imports from the segmentation phase's code or processed data — a deliberate trade-off accepted up front in exchange for genuine independence between the two pipelines.

### 1.2 The general path to `v2_clean_scripts`

The pipeline did not arrive at its current form in one step. It evolved through five stages, each triggered by a concrete finding from the stage before it:

**Stage 1 — Original 3-model pilot (6 combos).** ResNet18, MobileNetV3-Small, and EfficientNet-B0, each trained on two tissue crops (`palpebral`, `forniceal_palpebral`), frozen ImageNet backbone + trained linear head, Optuna-tuned learning rate/weight decay. Trained on Kaggle, results analyzed 2026-07-19. This run surfaced the finding that would drive nearly all subsequent work: **every one of the 6 models showed a lower AUC on India patients than Italy patients**, with `pos_weight`-driven class-imbalance effects the leading suspect.

**Stage 2 — 9-architecture expansion (18 combos, "v2").** To broaden the thesis's model coverage and standardize training protocol, the roster was expanded to 9 architectures (adding DenseNet121, ConvNeXt-Tiny, RegNetY-400MF as CNNs, and Swin-Tiny, ViT-B/16, ViT-L/16 as transformers), all retrained under one unified protocol (100-epoch ceiling, `dropout_rate` added as a third tuned Optuna hyperparameter after finding the original 3 models had inconsistent, partly hardcoded dropout). Evaluation was enriched with `sensitivity`/`specificity`/`balanced_accuracy` and stratified ROC plots. Run in two Kaggle batches by weight class; completed 2026-07-31.

**Stage 3 — Single-model deep dive (5-fold CV).** The best v2 combo (EfficientNet-B0/forniceal_palpebral) was subjected to a dedicated 5-fold cross-validation pipeline to test whether its result was reproducible. It confirmed the India/Italy gap was systematic, not a single-split artifact: **India recall was exactly 1.000 across all 5 independently-trained folds**, alongside India specificity of only 0.35±0.11 — strong evidence of a near-deterministic "predict anemic for India" shortcut.

**Stage 4 — Data bug discovery.** While building a Grad-CAM attention-visualization tool to investigate the shortcut directly, a genuine data-processing bug was found: `prepare_dataset.py` assumed every source crop encoded its tissue cutout via an alpha channel, and alpha-composited it onto black. A subset of source crops instead used an **opaque white background** with no usable alpha — the exact same convention issue the segmentation phase had independently discovered and fixed months earlier, but classification's from-scratch reimplementation had never received the equivalent fix. A full scan found **30 of 211 `forniceal_palpebral` patients affected — 100% of them Italy** — a country-correlated visual artifact that was itself a plausible shortcut cue, layered on top of the class-imbalance shortcut already suspected.

**Stage 5 — Clean-data retrain (`v2_clean_scripts`, the subject of this report).** The bug was fixed by porting the segmentation phase's already-validated detection/fallback logic (`_alpha_is_functional()`), the entire 217-patient dataset was reprocessed, and the fix was independently verified twice (428/428 images passing a stricter 4-check validation on the second pass). Rather than patch a subset, the project author decided on the comprehensive route: **retrain the full 18-combo v2 sweep from scratch against the corrected data**, so that every prior result set (original 6-combo, dirty-data 18-combo v2, 5-fold CV) stays on record as a historical baseline while a bug-free comparison becomes available for the first time. This decision is the direct origin of `v2_clean_scripts/`.

---

## 2. Actions Taken to Build and Execute `v2_clean_scripts`

### 2.1 Entry-point scripts (18 files)

`classification/v2_clean_scripts/train_{architecture}_{tissue_type}_v2_clean.py` — one thin script per (architecture, tissue type) combination, covering all 9 architectures × 2 tissue types. These were **mechanically generated**, not hand-transcribed, by parsing each existing `v2_scripts/train_*_v2.py` script's own `run_study()` call — reducing the risk of an 18-file manual transcription error. Every generated script was verified to `py_compile` cleanly, and correct `(architecture, tissue_type)` extraction was confirmed for all 18 pairs.

**Naming discipline:** every script's `model_name` carries the `_v2_clean` suffix, not a bare `_v2` re-run. This was a deliberate isolation choice — since `trainer_engine.py` derives every output path (`best_{model_name}.pth`, `{model_name}_*` logs/plots) from `model_name`, re-running under the old `_v2` name would have silently overwritten the dirty-data v2 results instead of producing a comparable, distinct third generation. The result is that three full generations of results (v1, dirty v2, clean v2) now coexist on disk under distinguishable filenames, with none overwritten.

### 2.2 Shared engine configuration

Two changes were made to the shared `trainer_engine.py`/`dataset.py` (inherited automatically by all 18 entry-point scripts, since none hardcode their own hyperparameters):

- **`TPESampler(seed=SEED, n_startup_trials=5)`**, reduced from Optuna's actual default of 10 (confirmed via direct inspection of the installed library, not assumed). With the project's fixed 12-trial-per-combo budget, the default would have spent 10 of 12 trials on pure random sampling before Optuna's Bayesian search engine ever engaged — leaving only 2 informed trials. Reducing to 5 gives a real 7-trial informed-search budget.
- **`BATCH_SIZE` raised from 16 to 32** in `dataset.py` — safe because every backbone is frozen (only a small linear head trains), giving comfortable headroom on Kaggle GPUs and halving optimizer steps per epoch across the sweep.

A third option — narrowing the learning-rate search range specifically for the three transformer architectures, on the classic "ViT fine-tuning is unstable at high LR" concern — was **considered and explicitly rejected**: that instability concern is about training self-attention weights directly, which does not apply here since every backbone (CNN or transformer) is frozen. Splitting the search space without pilot evidence also risked biasing the CNN-vs-transformer comparison itself. This was deferred to a possible narrower, evidence-based re-search scoped only to Phase 2's eventual champions.

### 2.3 Kaggle notebooks (2 notebooks, split by architecture family)

- **`classification-cnn-clean.ipynb`** — 12 combos (RegNetY-400MF, MobileNetV3-Small, EfficientNet-B0, ResNet18, DenseNet121, ConvNeXt-Tiny, cheapest-first, × 2 tissue types). Zips to `cnn_clean_results.zip`.
- **`classification-vit-clean.ipynb`** — 6 combos (Swin-Tiny, ViT-B/16, ViT-L/16, cheapest-first — ViT-L/16 queued last as the single most expensive combo in the entire 9-architecture roster at 304.3M parameters, × 2 tissue types). Zips to `vit_clean_results.zip`.

Both notebooks were generated by transforming the already-Kaggle-verified `classification-final-fixed.ipynb` template (reusing its proven Setup/Data cells), and both call `sync_outputs()` after every individual training cell — not just once at the end — so a mid-run Kaggle interruption never loses already-completed combos. Both were verified before being considered ready: `nbformat.validate()` passed, and every code cell was checked with `IPython.core.inputtransformer2.TransformerManager` (chosen specifically because a naive `compile()` check produces false positives on Jupyter's `!shell`/`%magic` cell syntax).

### 2.4 Results organization tooling (`organize_and_compare.py`)

A single script handles turning a downloaded Kaggle results zip into a structured, comparable record, and was written to be **batch-agnostic by design**: it discovers combo names dynamically from `outputs/logs/*_study_summary.json` filenames rather than a hardcoded architecture list, so the same script works unchanged whether it is pointed at the CNN batch, the transformer batch, or both merged together. For each discovered combo it:

1. Moves the checkpoint, 2 log files, and 4 plot files into a dedicated `outputs/{combo_name}/` subfolder (9 files per combo).
2. Extracts headline metrics (F1, Accuracy, Recall/Sensitivity, Precision, Specificity, Balanced Accuracy, AUC, and the India/Italy AUC gap) from each combo's `study_summary.json` into a per-combo `metrics_summary.json`/`.md`.
3. Builds a cross-combo ranked comparison in `outputs/model_comparison/` — `comparison_table.csv` (full detail) and `comparison_report.md` (formatted table plus a "top performer" / "best confound-handling" callout).

### 2.5 Execution timeline and debugging

**CNN batch (12 combos), completed 2026-08-02:** After the project author ran `classification-cnn-clean.ipynb` on Kaggle and placed the downloaded `cnn_clean_results.zip` at `outputs/`, the results were extracted, organized, and compared using the tooling above. Two `.gitignore` anchoring gaps were found and fixed during this step — patterns like `outputs/*/*.pth` are anchored to the directory the `.gitignore` file itself lives in and silently fail to match once checkpoints move to a deeper nested path; each gap was caught and fixed (verified via `git check-ignore -v`) before any checkpoint could be accidentally staged. A separate content-preservation check (hash comparison, byte-level `od -c` inspection to distinguish a real edit from a pure CRLF/LF line-ending artifact) was performed when the project author manually relocated the prior generation's dirty-data results to a sibling directory, confirming zero data loss. The organized CNN results and tooling were committed (`c6f03a0`), and a second, Kaggle-auto-pushed commit containing the fully executed notebook was reconciled via a cell-by-cell diff against origin before merging (`520e29f`).

**Transformer batch (6 combos), completed 2026-08-04 — the work performed in this session:**

1. **Extraction.** `vit_clean_results.zip` (3.10 GB, 45 entries) was inspected before extraction (entry count and top-level structure confirmed to match the CNN batch's flat `checkpoints/`/`logs/`/`plots/` layout) and then extracted via .NET's `ZipFile.ExtractToDirectory`. A PowerShell cmdlet incompatibility with the boolean overwrite parameter was hit and resolved by omitting the flag, since no overwrite was actually needed.
2. **Organization.** `organize_and_compare.py`'s dynamic discovery logic was exercised exactly as it was designed to be: run unmodified, it correctly found only the 6 newly-extracted combos (the 12 CNN combos had already been moved out of the flat staging directories in the prior step) and organized each into its own subfolder, matching the CNN batch's file layout exactly.
3. **Comparison merge, not overwrite.** A plain re-run of the comparison-building step would have processed only the 6 newly-organized combos and silently overwritten `model_comparison/` with an incomplete 6-row table. Instead, all 18 combo directories present on disk were scanned directly, metrics were re-extracted for the full set, and the comparison table/report were rebuilt to cover the complete 18-combo sweep.
4. **Two issues caught and fixed in passing:** a file lock (the project author had `comparison_table.csv` open in Excel, raising a `PermissionError` on write — resolved once closed) and a stale report title ("CNN Batch — Model Comparison," left over from when only CNN results existed) corrected to "Full 18-Combo Model Comparison" so the artifact accurately describes its own contents going forward.
5. **Git-safety check.** `git status` was used to confirm the `.gitignore` rule already added during the CNN-batch step correctly covers the new transformer checkpoints too — zero `.pth` files appeared as stageable, confirming no manual gitignore fix was needed this time.

---

## 3. Summary of Results

With both batches complete, `classification/v2_clean_scripts/outputs/model_comparison/` now holds a single, ranked comparison across all **18 (architecture, tissue-type) combinations**, trained on the bug-fixed dataset:

| Rank | Model | F1 | Balanced Acc. | AUC | India/Italy AUC Gap |
|---|---|---|---|---|---|
| 1 | ConvNeXt-Tiny / palpebral | 0.9333 | 0.9474 | **0.9398** | 0.1000 |
| 2 | ViT-B/16 / palpebral | 0.9333 | 0.9474 | 0.9098 | 0.1333 |
| 3 | EfficientNet-B0 / forniceal_palpebral | 0.9032 | 0.9118 | 0.8824 | 0.3365 |
| 4 | ViT-L/16 / palpebral | 0.8966 | 0.9117 | 0.9173 | 0.0583 |
| 5 | RegNetY-400MF / forniceal_palpebral | 0.8966 | 0.9055 | 0.8739 | 0.4500 |
| … | *(13 further combos)* | | | | |
| 18 | MobileNetV3-Small / forniceal_palpebral | 0.7568 | 0.7353 | 0.7941 | **0.0192** |

*(Full 18-row table: `classification/v2_clean_scripts/outputs/model_comparison/comparison_report.md`.)*

**Top performer:** ConvNeXt-Tiny/palpebral and ViT-B/16/palpebral are **tied on F1 (0.9333) and Balanced Accuracy (0.9474)** — a CNN and a transformer reaching the identical headline score. ConvNeXt-Tiny is the single best overall result on balance: it leads on AUC (0.9398 vs. 0.9098) and has the smaller India/Italy confound gap (0.1000 vs. 0.1333).

**Best confound-handling:** MobileNetV3-Small/forniceal_palpebral retains the smallest India/Italy AUC gap of all 18 combos (0.0192), continuing to trade that robustness for a notably weaker F1 (0.7568) — the same "best-overall and best-confound-handling disagree" pattern observed in every prior sweep of this project (the original 6-combo run, the dirty-data 18-combo v2 run, and now the clean 18-combo run).

**A pattern worth flagging for Phase 2:** beyond the single MobileNetV3-Small outlier, the next three smallest confound gaps in the full 18-combo table all belong to transformer architectures on the palpebral crop — Swin-Tiny (0.0500), ViT-L/16 (0.0583), and (with a larger gap but still relatively favorable) ViT-B/16 (0.1333). This is a candidate signal, not a statistically settled finding given the small sample (6 transformer combos vs. 12 CNN combos), but it is consistent enough to weigh when Phase 2 selects the top-2 CNN and top-2 transformer champions for the next stage (3-fold cross-validation).

**What this milestone unblocks:** Phase 1 of the project author's 3-phase plan (18-combo Optuna sweep → Phase 2: select top-2 CNN + top-2 transformer → Phase 3: 3-fold CV on the 4 champions) is now complete. Phase 2 selection can proceed on a fully bug-free, fully populated 18-combo dataset for the first time in the project's history.

**Open items, not yet done:**
- A direct, paired comparison against the *dirty-data* v2 18-combo results (same architectures, same tissue types, pre-fix images) has not yet been performed — this is the analysis that would quantify how much of the historical India/Italy gap was attributable to the white-background artifact specifically, versus the class-imbalance/`pos_weight` mechanism identified earlier.
- The `.project_memory` updates recording this milestone, and the newly organized output files themselves, are staged in the working tree but **not yet committed or pushed** — pending a separate, explicit go-ahead per this project's standing git-discipline rule.
- Phase 2 champion selection has not yet started.

---

*This report is grounded entirely in `classification/.project_memory/01_roadmap.md`, `02_current_status.md`, and `05_kaggle_training_phase.md`, and in the executed output files under `classification/v2_clean_scripts/outputs/`. No figures in this report were estimated or inferred beyond what those sources record.*
