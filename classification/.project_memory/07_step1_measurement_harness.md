# Step 1 — Measurement Harness (classification/, Phase 4)

**Started:** 2026-08-04. Own file per `03_tech_stack_and_rules.md` rule #9.

**Status: code built and verified; the real 12-combo run has NOT been executed.** Structural verification passes for real, and a full end-to-end dry run (2 epochs, 2 repeats) exercised every code path — but no production numbers exist yet.

---

## 1. Why this phase exists

The project author asked for a defensibility framework for the trained classifiers — proof they learn physiological signal rather than country-correlated shortcuts — initially scoped around Grad-CAM. Analysis of the actual `v2_clean_scripts/outputs/model_comparison/comparison_table.csv` turned up a prior problem that had to be solved first: **the metric being used to detect the confound is itself statistically underpowered.**

The single 70/15/15 split leaves 33 validation patients. Composition, computed from `splits.csv`, not assumed:

| | anemic | healthy | discordant pairs |
|---|---|---|---|
| India | 10 | 4 | **40** |
| Italy | 4 | 15 | 60 |

**Independent confirmation the pair count is right:** every one of the 12 India AUC values in `comparison_table.csv` is an exact multiple of 1/40 = 0.025 (checked programmatically, all 12). That can only happen with 10×4 pairs.

Consequences:
- India AUC 95% CI half-width ≈ **±0.27** (Hanley–McNeil, n_pos=10 / n_neg=4).
- The full observed India AUC spread across the 12 combos (0.550 → 1.000) sits inside that noise band.
- The distance between the "worst" (0.550 = 22/40 pairs) and a "moderate" combo (0.750 = 30/40) is **8 pairs out of 40** — with only 4 healthy India patients, each participating in 10 pairs, that is arithmetically about **one patient's rank**.

So no intervention in any later step could have been shown to work. Fixing the measurement had to come before fixing the model.

## 2. What IS real in the existing results (verified, keep citing these)

Underpowered per-model rankings do not mean nothing is there. Computed directly from the 12-combo table:

- **Systematic country gap is solid.** Mean India AUC **0.7104** vs mean Italy AUC **0.9370**; India < Italy in **11 of 12** models (two-sided sign test **p = 0.0064**). The population-level effect replicates across 12 independent trainings even though individual values are noisy.
- **Tissue type outperforms architecture as an explanatory axis.** In paired comparisons (same architecture, same split), palpebral beats forniceal_palpebral on India AUC in **5 of 6** cases, mean **+0.121** (mean India AUC: palpebral 0.771 vs forniceal 0.650). The project author's earlier severe/moderate/excellent grouping of the top 5 largely tracks this tissue axis, not the architecture.
- **The mechanism, quantified.** Overall AUC decomposes exactly as `0.150 × AUC_India + 0.226 × AUC_Italy + 0.624 × AUC_cross-country`. **62.4% of the pairs driving the headline AUC are cross-country**, and 150 of those 166 (90%) are India-anemic vs Italy-healthy — the direction where the country prior alone gives the right answer. Working the decomposition backwards for RegNetY-400MF/forniceal (India 0.550, Italy 1.000, overall 0.874) yields a cross-country AUC of 0.918: excellent at the 62% a country detector solves for free, at chance on the 15% requiring per-patient physiology.
- **Dataset factors behind it** (computed from `splits.csv`, all 217 patients): India-healthy is only **19 of 151** training patients (12.6%) — the smallest cell that matters, since India AUC depends entirely on ranking India-healthy below India-anemic. And India-healthy patients sit a mean of **1.16 g/dL** above their WHO cutoff with **56% within 1.0 g/dL**, versus Italy's 1.89 g/dL and 18%. Part of the India gap is therefore **intrinsic task difficulty, not shortcut learning** — a distinction Step 2 must separate, because debiasing cannot fix the intrinsic share.

## 3. Two corrections recorded so they are not repeated

1. **The "recall 1.000 / specificity 0.35 across 5 folds" statistic does NOT belong to the `v2_clean` models.** It comes from `06_efficientnet_b0_5fold_cv_deep_dive.md`'s pipeline — a different combo, k-fold protocol, and pre-fix (dirty) data. It was mistakenly cited as evidence about the v2_clean CNNs during this session. Do not carry it across.
2. **There is no blanket "predict anemic" bias in the v2_clean CNNs — the specificity column rules it out.** Six models (not five) show recall = 1.000, and their specificities are 0.895 / 0.824 / 0.789 / 0.706 / 0.529 / 0.471 — the two best performers pair perfect recall with 0.89 and 0.82 specificity. A degenerate always-anemic model on this val set would score specificity 0.000, F1 0.596, balanced accuracy 0.500; every one of the 12 is comfortably clear of that. The confound case rests on the **AUC** evidence (§2), which is threshold-independent, not on a class-bias claim.

## 4. The approved 6-step roadmap

| Step | Objective | Gate |
|---|---|---|
| **1** | **Measurement harness** — pooled out-of-fold CV + bootstrap CIs. No model changes. | India AUC CI half-width ≤ 0.12; negative control passes; baseline frozen |
| 2 | Diagnostics — country linear probe, leave-one-country-out, Hgb-severity-stratified AUC | Shortcut magnitude quantified; intrinsic-difficulty share estimated |
| 3 | Loss/sampling interventions — 4-cell (country×label) reweighting, Group-DRO | Paired ΔIndia-AUC CI excludes 0 |
| 4 | Image-space interventions — CLAHE/HSV normalization, grayscale ablation | Same paired-Δ gate |
| 5 | Partial unfreezing (last stage, discriminative LR, worst-group early stopping) | Paired Δ gate + no worst-group regression |
| 6 | Sealed test set, evaluated exactly once | — |

**Sequencing rationale (data interventions before unfreezing):** with 184 patients against 11M–304M backbone parameters, full fine-tuning would overfit and — the more specific risk — a higher-capacity trainable network can encode country *better*, since country is more learnable than pallor here. Steps 3–4 are cheap, reversible, and do not add capacity for the model to memorize country. Step 5 is still expected to be necessary eventually (a frozen ImageNet representation may simply not contain a pallor direction), just not first. Domain-adversarial methods (DANN) are near-pointless with a frozen backbone and are sequenced after Step 5.

## 5. What was built — `classification/step1_cv_harness/`

Fresh module, project-author-directed clean slate. **It does not reuse or extend `datapreparepipeline/efficientnet_b0_forniceal_5fold_cv/`, which is deprecated** (trained on pre-fix data; the author intends to delete it).

| File | Role |
|---|---|
| `cv_config.py` | Design constants, gate thresholds, combo discovery + locked hyperparameters |
| `cv_data.py` | Pool construction, fold building, label-shuffle control, dataset |
| `cv_engine.py` | Per-fold training → out-of-fold prediction |
| `cv_stats.py` | AUC, bootstrap CIs, `paired_delta_auc()` for Steps 3–5 |
| `validate_harness.py` | Structural verification (checkpoints 1–4, 6) — trains nothing |
| `run_cv_harness.py` | Runner, one combo per invocation |
| `aggregate_baseline.py` | Gate evaluation (5, 7, 8) + frozen baseline artifact (9) |
| `README.md` | Design rationale and execution steps |

Plus `classification/Kaggle-Notebook/step1-cv-harness.ipynb` (34 cells; Setup/Data reused verbatim from the Kaggle-proven `classification-cnn-clean.ipynb`).

**Design.** 5-fold × 5-repeat repeated stratified CV over the train+val pool, stratified on the compound `country × label` key (never the label alone — India-healthy is 23 in the pool, and label-only stratification could hand a fold 2 of them). Predictions are pooled **out-of-fold within each repeat** — every patient holds out exactly once, so AUC is computed once over the full pooled vector rather than as a mean of 5 noisy per-fold AUCs.

**The live pipeline's architecture builders and transforms are imported, not copied**, so the models measured are bit-identical to what v2_clean trained. Locked hyperparameters are parsed out of each combo's own `*_study_summary.json` rather than hand-transcribed. No re-tuning happens in Step 1 — that would confound "better measurement" with "better hyperparameters."

**Precision gained** (measured by `validate_harness.py`, not estimated):

| | pool n | India pairs | Italy pairs |
|---|---|---|---|
| Single 70/15/15 split | 33 val | 40 | 60 |
| Pooled OOF, palpebral | 184 | **1,311** (57×23) | **1,680** (20×84) |
| Pooled OOF, forniceal_palpebral | 178 | **1,311** (57×23) | **1,501** (19×79) |

**New empirical fact:** all 6 patients missing a forniceal crop are **Italy**, so the India cells — and India pair count — are identical across both tissue types. Computed, not assumed; checkpoint 4 deliberately does not hardcode expected counts for forniceal.

**Honest limit, stated up front:** 23 India-healthy patients is irreducible without new data. Expect a CI half-width around ±0.10–0.12, not ±0.03. Improvements of roughly **0.15+ become detectable; 0.05 does not.** Gate 8 is set at 0.12 accordingly. If it fails, Steps 3–5's success criteria must be renegotiated rather than quietly ignored.

## 6. Two deliberate deviations from the v2_clean protocol (both corrections)

1. **Early stopping runs on an inner split** carved from each fold's own training portion (15%, itself stratified on the 4-cell key), never on the outer held-out fold. v2_clean early-stopped on the same split it then reported, which selects the reported epoch using the evaluation data — a real optimistic bias. Checkpoint 2 asserts the inner split never touches the outer fold. `pos_weight` is likewise computed from the inner-training portion only.
2. **No per-fold checkpoints are written.** Step 1 needs out-of-fold predictions, not weights; 25 checkpoints × 12 combos would be tens of GB of Kaggle output nothing downstream reads.

Everything else — frozen backbone, head shape, augmentation, loss, optimizer, 250-epoch ceiling, patience 7, batch size 32 — matches v2_clean exactly.

## 7. Nine verification checkpoints

| # | Check | Where | Status |
|---|---|---|---|
| 1 | Sealed test set absent from every fold | `validate_harness.py` | **PASS** (0 of 33 in any fold, both tissue types) |
| 2 | Val folds partition the pool exactly once per repeat; inner split never touches outer fold | `validate_harness.py` | **PASS** |
| 3 | ≥3 India-healthy and ≥3 Italy-anemic per outer val fold | `validate_harness.py` | **PASS** (observed minima 4 and 3) |
| 4 | Pair counts computed and reported (not assumed for forniceal) | `validate_harness.py` | **PASS** (§5 table) |
| 5 | Label-shuffle negative control | `run_cv_harness.py --shuffle-control` | passed in dry run; **not yet at full scale** |
| 6 | Same seed → identical fold assignments | `validate_harness.py` | **PASS** |
| 7 | Plausibility: pooled India AUC within 0.40–0.90 | `aggregate_baseline.py` | untested (needs the real run) |
| 8 | **Precision: India AUC CI half-width ≤ 0.12** | `aggregate_baseline.py` | untested at full scale |
| 9 | Frozen baseline artifact with full provenance | `aggregate_baseline.py` | mechanism verified in dry run |

Checkpoints 1–4 are **re-run inside `run_cv_harness.py` immediately before training**, so a run launched with different `--folds/--repeats/--seed` cannot silently skip the geometry that was actually verified.

**Negative control design (checkpoint 5).** Two modes. `within_country` permutes labels inside each country, preserving each country's label rate while destroying every per-patient association: per-country AUC **must** collapse to 0.50 (the leakage test), while overall AUC may legitimately stay above 0.50 — which is a direct empirical demonstration of §2's shortcut mechanism. `global` destroys the country-label association too, so every AUC must reach chance. Given this project's history with silent data bugs (white-background convention; v1 template matching), this is the check least worth skipping.

## 8. Verification actually performed (2026-08-04)

- **Structural checks: all pass for real** on both tissue types, exit code 0.
- **Full end-to-end dry run** — 2 epochs, 2 repeats, MobileNetV3-Small/palpebral, artifacts written to a throwaway scratch directory and deleted afterward (standard convention). Exercised 10 fits → out-of-fold predictions → pooled AUC → bootstrap → aggregation → gates. Zero errors.
- **Negative control passed in that dry run**, for real: with labels permuted within country, India AUC CI came out [0.279, 0.564] and Italy [0.293, 0.538] — both covering 0.50.
- Gate 8 correctly **failed** in the dry run (2-epoch models, 2 repeats), confirming the gate is live rather than vacuous.

**Two real bugs found and fixed during verification:**

1. **BatchNorm running statistics are not frozen by `requires_grad=False`.** `running_mean`/`running_var` keep updating in `train()` mode in 5 of the 6 CNNs. The original best-epoch snapshot captured only head parameters, which would have paired the best epoch's head with the **last** epoch's normalization statistics — not the model that achieved the best inner-validation loss. The snapshot now includes all buffers. (Worth noting for the thesis: "the backbone is frozen" is not strictly true for BN statistics in any of this project's CNN runs, including v2_clean's.)
2. The paired-bootstrap alignment guard emitted one merged message for two distinct failure causes (different patients vs. different labels) and pointed at the wrong one when hit during testing. Now split into three specific diagnostics.

**Paired vs unpaired comparison, measured on real out-of-fold predictions** with a synthetic probability shift — this validates the core design decision for Steps 3–5:

| Shift | Paired ΔIndia AUC [95% CI] | Detects? | Unpaired ΔIndia AUC [95% CI] | Detects? |
|---|---|---|---|---|
| 0.00 | +0.000 [0.000, 0.000] | no (correct) | +0.001 [−0.144, +0.154] | no (correct) |
| 0.03 | +0.154 [+0.073, +0.252] | **yes** | +0.150 [−0.043, +0.340] | **no** |
| 0.10 | +0.305 [+0.146, +0.523] | yes | +0.297 [+0.044, +0.562] | yes |

## 9. Binding contract for Steps 3–5

Every later intervention **must** reuse the fold assignments persisted in each combo's `fold_manifest.json`, then compare against the frozen baseline with `cv_stats.paired_delta_auc()` — **not** by checking whether two independent CIs overlap. Per the table above, with 23 India-healthy patients the unpaired comparison cannot detect a realistic intervention effect. This is why fold assignments are returned and persisted as data rather than being a side effect.

## 10. Split into two Kaggle notebooks (2026-08-06)

Project-author decision: run the real Step 1 measurements as **two separate notebooks**, one per architecture family, mirroring how `classification-cnn-clean.ipynb`/`classification-vit-clean.ipynb` split the original v2_clean training — the ViT roster (`vit_l_16` = 304.33M frozen backbone params) is far more compute-expensive per fit even with the backbone frozen, since every fit still pays the full forward-pass cost through the network 25 times.

**Prerequisite done first: the ViT/Swin v2_clean batch results were organized and committed.** They had already been placed into per-combo folders locally (9 files each, matching the CNN combo structure — checkpoint, study summary, trials CSV, 4 plots, metrics summary) and `model_comparison/comparison_table.csv` already correctly reflected all 18 combos, but none of it had been committed. Verified complete (file counts, a spot-checked metrics value against its own combo's row) and committed. 36 leftover duplicate files (a flat `logs/`/`plots/` extraction remnant plus `.rar` archives of them) were byte-compared against the organized combo folders — confirmed identical — and deleted.

| Notebook | Combos | Output zip |
|---|---|---|
| `step1-cv-harness-cnn.ipynb` (renamed from `step1-cv-harness.ipynb`) | 12 CNNs | `step1_cv_results_cnn.zip` |
| `step1-cv-harness-vit.ipynb` (new) | 6 transformers | `step1_cv_results_vit.zip` |

The negative control (checkpoint 5) lives **only** in the CNN notebook — it tests fold construction and pooling, not architecture-specific behavior, so it isn't duplicated. Both notebooks' `aggregate_baseline.py` cell produces a valid *partial* baseline (12-combo or 6-combo) when run standalone; the CNN notebook's own aggregate step will show gate 5 passing (its own control), the ViT notebook's will correctly show gate 5 failing with `n_controls_run: 0` until merged. **The real 18-combo baseline requires extracting both zips' `outputs/` into the same local `classification/step1_cv_harness/outputs/` and running `aggregate_baseline.py` once** — combo discovery is dynamic, so no code change is needed for that step.

## 11. CNN notebook run once (2026-08-08); design gap found; harness extended and requires a re-run

The CNN notebook (`step1-cv-harness-cnn.ipynb`) was run for real on Kaggle and completed (~4h45m, 11:11→15:57) — 12 combos + the within-country/global negative controls, downloaded to `classification/v2_clean_scripts/outputs/Kfold_output/CNN/`. This is the first real production run of the harness.

**Gap found on inspection:** the project author asked to see, per model, a loss curve and confusion matrix per fold, plus an overall-evaluation view and a cross-combo comparison folder — none of which existed. The original design (`cv_config.py`, deviation 2) deliberately saved *only* `oof_predictions.csv`/`cv_metrics.json`/`fold_manifest.json` — no plots, and critically, **no per-epoch loss history was ever recorded**: `cv_engine.py`'s training loop computed a fresh train/val loss every epoch but only ever compared it against the running best, discarding each epoch's value immediately. This is not reconstructable after the fact from what was already downloaded — the CNN run's results do not contain it.

**Harness extended, verified via local dry run, not yet re-run for real:**
- `cv_engine.py`: `run_fold()` now appends every epoch's train loss and inner-validation loss to two history lists and returns them (cheap — a couple of floats per epoch).
- `cv_stats.py`: new `fold_level_metrics()` — confusion matrix, ROC curve, and the full accuracy/precision/recall/specificity/balanced_accuracy/F1/AUC set, computed on **one fold's own ~37-patient held-out set** (a per-fold diagnostic, explicitly not the pooled headline statistic `bootstrap_auc_cis()` already computes).
- New `cv_plots.py` — all plotting, kept separate from the numeric modules. Four per-combo grid plots (loss curves, confusion matrices, ROC curves, all laid out as an n_repeats×n_splits grid so every fold is visible in one image, plus a fold-metrics-summary line chart) and two cross-combo comparison plots (country AUC with CI whiskers, India/Italy gap with CI whiskers).
- `run_cv_harness.py`: collects histories + per-fold metrics during the training loop, saves `fold_metrics.csv` + `fold_diagnostics.json` per combo, calls the four plotting functions into `outputs/{combo}/plots/`.
- `aggregate_baseline.py`: writes a new `outputs/comparison/` folder (sibling to `outputs/baseline/`, per explicit request to keep it separate) with the two cross-combo plots, built from the merged baseline DataFrame.
- Both notebooks' "Done" markdown updated to describe the new artifacts; no notebook *code* cells needed changes, since they invoke the harness scripts generically and the new outputs are picked up automatically.

**Verified before calling it done:** a local dry run (2 combos, `MAX_EPOCHS` monkey-patched to 3, 2 repeats) ran the full path end-to-end with zero errors, and all 6 plot types (4 per-combo × 2 combos, plus 2 comparison plots) were visually inspected, not just checked for file existence — confirmed correct labeling, correct per-fold sample counts, AUC values on the ROC grid matching the confusion-matrix grid, and CI-whisker/pass-fail coloring rendering correctly on the comparison plots.

**Consequence: the already-completed CNN run's local results are missing the new artifacts and must be regenerated.** Since the code change is in the shared `step1_cv_harness/` module (not notebook-specific), both notebooks need a fresh Kaggle run.

**Correction, discovered while committing the fix (2026-08-08):** the paragraph above originally said the ViT notebook "hasn't run at all yet." That was wrong — while this fix was being built locally, **both** notebooks had actually already been executed on Kaggle and auto-pushed back to GitHub (`acee9e4` "Kaggle Notebook | step1-cv-harness-cnn | Version 1", `d78e592` "...step1-cv-harness-vit | Version 1"), discovered via `git fetch` immediately before pushing. Both runs completed all combos (12 CNN + 2 negative controls; 6 transformer) — but **both ran under the pre-fix code**, so both sets of results have the same gap: no loss curves, no confusion matrices, no ROC grids, no `comparison/` folder. Merged cleanly (no conflict — the executed notebooks only touched code-cell outputs/execution_count, this session's changes only touched markdown text), verified both notebooks still `nbformat.validate()` and the markdown edits survived, then pushed (`88acf44` the fix, merge `99a5872`).

## 12. Committed, pushed, and re-run launched (2026-08-08)

- Fix committed as `88acf44` (10 files: `cv_engine.py`, `cv_stats.py`, new `cv_plots.py`, `run_cv_harness.py`, `aggregate_baseline.py`, both notebooks' docs, this memory file, README). Staged by explicit filename, not `git add -A` — the working tree also had substantial unrelated, uncommitted `Segmentation/` changes (deletions, new files, modified memory) from outside this session that were deliberately left untouched.
- Merged the two executed-notebook commits (`99a5872`), pushed, confirmed local/remote in sync.
- **Project author pulled and relaunched both notebooks (Save & Run All) under the fixed code — confirmed in progress as of this entry.** This is now the run that should actually produce loss-curve/confusion-matrix/ROC grids and the per-notebook `comparison/` folders.

## 13. Not yet done

- **Both notebooks' re-run (started 2026-08-08) has not been confirmed complete.** Once both finish: download both zips, extract into the same local `classification/step1_cv_harness/outputs/`, run `aggregate_baseline.py` once for the real 18-combo baseline + merged `comparison/` charts.
- Gates 5 (at full scale), 7, and 8 are untested against production numbers. **Step 2 must not start until that merged run reports Step 1 clear.**
- Grad-CAM itself is **not** part of Step 1. The conclusion from the design discussion: with a frozen backbone and a `GAP → Dropout → Linear` head, Grad-CAM degenerates to exact CAM (channel weights *are* the learned head weights) — mathematically exact, but **spatially resolved and cue-blind**. A colour/illumination shortcut appears as a channel reweighting, not a spatial displacement, so a perfect Grad-CAM result is fully compatible with a total colour shortcut. Inputs are already tissue-isolated crops on black, so "does it look at the conjunctiva?" is largely answered by construction. Grad-CAM is therefore demoted to a qualitative supporting exhibit; Steps 2–4 carry the quantitative argument.
