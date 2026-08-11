# Step 1 — Measurement Harness

Pooled out-of-fold repeated stratified cross-validation for the 12 clean-data CNN combos.

**This step changes no model and applies no intervention.** It replaces a statistically unusable
estimator with a usable one and produces the frozen baseline that Steps 2–5 are measured against.

---

## Why this exists

The single 70/15/15 split leaves 14 India validation patients — 10 anemic and 4 healthy — so India
AUC is computed over 10 × 4 = **40 discordant pairs**. Consequences:

- Every India AUC in the v2_clean results is an exact multiple of 1/40 = 0.025 (confirming the pair count).
- The 95% CI half-width is roughly **±0.27**.
- The full observed spread across the 12 combos (India AUC 0.550 → 1.000) sits inside that noise band.
  The gap between the "worst" and "best" combo is 8 pairs out of 40 — arithmetically, about **one
  India-healthy patient's rank**.

No intervention in Steps 3–5 could be shown to work against that noise floor.

## What it produces

5-fold × 5-repeat CV over the 184-patient train+val pool, stratified on the compound `country × label`
key, with predictions pooled out-of-fold:

| | India pairs | Italy pairs |
|---|---|---|
| Single 70/15/15 split | 40 | 60 |
| Pooled out-of-fold (palpebral) | **1,311** (57 × 23) | **1,680** (20 × 84) |
| Pooled out-of-fold (forniceal_palpebral) | **1,311** (57 × 23) | **1,501** (19 × 79) |

## Per-combo artifacts (`outputs/{combo}/`)

The pooled statistic (`cv_metrics.json`, `oof_predictions.csv`) was the original, deliberately lean
design. Added after the first CNN run made clear that per-fold visual diagnostics were also wanted:

| File | Contents |
|---|---|
| `fold_metrics.csv` | One row per (repeat, fold): accuracy/precision/recall/specificity/balanced_accuracy/F1/AUC, computed on that fold's own ~37-patient held-out set (not the pooled 184) |
| `fold_diagnostics.json` | The raw per-fold loss histories + confusion matrices + ROC curves behind the plots below |
| `plots/{combo}_loss_curves_grid.png` | Train vs. inner-validation loss, one subplot per fold, laid out as an n_repeats × n_splits grid so every fold is visible in one image |
| `plots/{combo}_confusion_matrices_grid.png` | Same grid layout, one confusion matrix per fold |
| `plots/{combo}_roc_curves_grid.png` | Same grid layout, one ROC curve (with AUC) per fold |
| `plots/{combo}_fold_metrics_summary.png` | AUC/F1/Balanced Accuracy plotted across all (repeat, fold) units in one chart, for at-a-glance fold-to-fold consistency |

**These are per-fold diagnostics, not the headline number.** Each fold's own held-out set is only
~37 patients — exactly the small-N regime this harness exists to move away from as the *reported*
result. Use them to sanity-check individual folds (e.g. spot a fold that collapsed), not to cite a
per-fold AUC as "the" result — that's what the pooled `cv_metrics.json` is for.

Still **no model checkpoints** — Step 1 needs predictions and diagnostics, not weights.

## Cross-combo comparison (`outputs/comparison/`)

Written by `aggregate_baseline.py` alongside `outputs/baseline/`, as a separate folder:

- `country_auc_comparison.png` — India/Italy/overall pooled AUC, one bar per combo, 95% bootstrap CI whiskers, all three panels sorted by India AUC for cross-reference.
- `india_italy_gap_comparison.png` — the India−Italy AUC gap per combo with its own CI, green where the CI excludes 0.

All 6 patients lacking a forniceal crop are Italy, so the India cells are identical across tissue types.

**Honest limit:** 23 India-healthy patients is irreducible without new data. Expect a CI half-width
around ±0.10–0.12, not ±0.03. That makes improvements of roughly 0.15+ detectable — it does **not**
make 0.05 detectable. The gate is set accordingly.

---

## Files

| File | Role |
|---|---|
| `cv_config.py` | Design constants, gate thresholds, combo discovery + locked hyperparameters |
| `cv_data.py` | Pool construction, fold building, label-shuffle control, dataset |
| `cv_engine.py` | Per-fold training and out-of-fold prediction — also tracks per-epoch loss history |
| `cv_stats.py` | Pooled AUC + bootstrap CIs, `paired_delta_auc()` for Steps 3–5, and `fold_level_metrics()` (per-fold confusion matrix/ROC/full metric set) |
| `cv_plots.py` | All plotting — per-combo fold grids, and cross-combo comparison charts |
| `validate_harness.py` | Structural verification (checkpoints 1–4, 6) — trains nothing |
| `run_cv_harness.py` | Runner, one combo per invocation |
| `aggregate_baseline.py` | Gate evaluation (5, 7, 8), the frozen baseline artifact (9), and cross-combo comparison plots |

The live pipeline's architecture builders and transforms are **imported**, not copied, so the models
measured here are bit-identical to the ones v2_clean trained. The deprecated
`datapreparepipeline/efficientnet_b0_forniceal_5fold_cv/` is not referenced anywhere.

## Deliberate deviations from the v2_clean protocol

1. **Early stopping runs on an inner split** carved from each fold's own training portion, never on
   the outer held-out fold. v2_clean early-stopped on the same split it then reported, which selects
   the reported epoch using the evaluation data — a real optimistic bias. Step 1's purpose is a
   defensible number, so it is removed rather than reproduced.
2. **No per-fold checkpoints.** Step 1 needs predictions, not weights.

Everything else — frozen backbone, head shape, augmentation, loss, `pos_weight`, optimizer, epoch
ceiling, patience — matches v2_clean exactly.

---

## Verification checkpoints

| # | Check | Where |
|---|---|---|
| 1 | Sealed test set absent from every fold | `validate_harness.py` |
| 2 | Val folds partition the pool exactly once per repeat; inner split never touches the outer fold | `validate_harness.py` |
| 3 | ≥3 India-healthy and ≥3 Italy-anemic per outer val fold | `validate_harness.py` |
| 4 | Pair counts computed and reported (not assumed for forniceal) | `validate_harness.py` |
| 5 | Label-shuffle negative control | `run_cv_harness.py --shuffle-control` |
| 6 | Same seed → identical fold assignments | `validate_harness.py` |
| 7 | Plausibility: pooled India AUC within 0.40–0.90 | `aggregate_baseline.py` |
| 8 | **Precision: India AUC CI half-width ≤ 0.12** — the pass/fail criterion | `aggregate_baseline.py` |
| 9 | Frozen baseline artifact with full provenance | `aggregate_baseline.py` |

Checkpoints 1–4 are re-run inside `run_cv_harness.py` immediately before training, so a run launched
with different `--folds/--repeats/--seed` cannot silently skip the geometry that was verified.

---

## Execution — Kaggle

**Two notebooks, run separately, results merged locally afterward:**

| Notebook | Combos | Output zip |
|---|---|---|
| `classification/Kaggle-Notebook/step1-cv-harness-cnn.ipynb` | 12 CNNs | `step1_cv_results_cnn.zip` |
| `classification/Kaggle-Notebook/step1-cv-harness-vit.ipynb` | 6 transformers (`swin_t`, `vit_b_16`, `vit_l_16` × 2 tissue types) | `step1_cv_results_vit.zip` |

Split by architecture family for the same reason the original v2_clean training was split this way
(`classification-cnn-clean.ipynb` / `classification-vit-clean.ipynb`): the transformer roster, especially
`vit_l_16` at 304.33M frozen backbone parameters, is far more compute-expensive per fit — even with the
backbone frozen, every one of the 25 fits still pays the full forward-pass cost through the whole network.
The ViT notebook is very likely the single most expensive run in this project to date.

**The label-shuffle negative control (checkpoint 5) runs only in the CNN notebook**, against
`mobilenet_v3_small_palpebral_v2_clean`. It is not architecture-specific — it tests the harness's fold
construction and pooling logic, which is identical regardless of which model runs through it — so it is
not duplicated in the ViT notebook.

Steps, per notebook:

1. Push this module to GitHub first — each notebook's `git clone` needs it there.
2. Run cell 4 (`/kaggle/input` listing) and confirm the data-copy cell's `SRC_DIR` matches the real mount path.
3. Save Version → Save & Run All. `sync_outputs()` runs after every combo, so an interrupted session
   still yields a complete zip of what finished.
4. Download the notebook's output zip.

Budget fallback if a session is tight: add `--repeats 3` to that notebook's training cells.

**After both notebooks finish:** extract both zips' `outputs/` into the same local
`classification/step1_cv_harness/outputs/`, then run `aggregate_baseline.py` once locally. Combo
discovery is dynamic (globs `outputs/*/cv_metrics.json`), so no code change is needed to go from a
12- or 6-combo partial baseline to the full 18-combo one — and only the merged run will show gate 5
passing, since the control only exists in the CNN notebook's output.

## Execution — local

```bash
python classification/step1_cv_harness/validate_harness.py
```

```bash
python classification/step1_cv_harness/run_cv_harness.py --list
```

```bash
python classification/step1_cv_harness/run_cv_harness.py --all
```

```bash
python classification/step1_cv_harness/run_cv_harness.py --combo mobilenet_v3_small_palpebral_v2_clean --shuffle-control within_country
```

```bash
python classification/step1_cv_harness/aggregate_baseline.py
```

---

## Post-baseline: Italy threshold recalibration (`threshold_recalibration.py`)

Gate 8 only checks AUC precision; it says nothing about precision/recall/F1 at the conventional 0.5
threshold. A fresh, country-stratified computation of those (pooled per-repeat like the AUC estimator)
showed Italy's F1 well below India's across all 18 combos despite Italy's AUC being consistently
*higher* — the mechanism is base-rate sensitivity (Italy's pool is ~19% anemic, so a fixed 0.5 cutoff
sits in the wrong place), not a discrimination problem. `threshold_recalibration.py` tests whether
recalibrating Italy's decision threshold (only) fixes this, using **nested leave-one-fold-out**
threshold selection on the existing `oof_predictions.csv` — no retraining, no GPU.

**Result: mostly negative.** Only 6/18 combos improved Italy F1 under honest nested evaluation (median
ΔF1 −0.017). A naive, non-nested version of the same threshold search (select and evaluate on the same
Italy data) makes it look like all 18/18 combos improve — a documented negative control quantifying how
much of the apparent gain is overfitting to which ~104 Italy patients are in the pool, not a real,
generalizable effect. See `outputs/threshold_recalibration/italy_threshold_recalibration.md` and
`outputs/step1_italy_threshold_recalibration.xlsx` for the full per-combo breakdown, threshold stability
(mean/SD across the 25 nested selections), and the naive-vs-nested comparison.

```bash
python classification/step1_cv_harness/threshold_recalibration.py
```

**Exact best-F1 threshold leaderboard (`best_threshold_leaderboard.py`).** A narrower companion: not
"does this generalize" (above), just "what is the exact F1-maximizing threshold for Italy, given every
prediction available." Searches every midpoint between consecutive sorted predicted probabilities
(the true optimum, not a 0.01 grid) on all of Italy's pooled out-of-fold predictions. India stays fixed
at 0.5; Overall combines both under that mixed policy; AUC columns are unchanged (threshold-independent).
This is a deployment-style number, not a validated estimate — pair it with the honest nested result above
before trusting it to generalize. Output: `outputs/best_threshold/italy_best_threshold_exact.{csv,json}`,
`outputs/step1_leaderboard_best_threshold.xlsx`.

```bash
python classification/step1_cv_harness/best_threshold_leaderboard.py
```

## Contract for Steps 3–5

Every later intervention **must** reuse the fold assignments persisted in each combo's
`fold_manifest.json`, then compare against this baseline with `cv_stats.paired_delta_auc()` — not by
checking whether two independent CIs overlap.

This is not a stylistic preference. Measured on real out-of-fold predictions with a synthetic +0.03
probability shift:

| Comparison | ΔIndia AUC | 95% CI | Detects the effect? |
|---|---|---|---|
| Paired | +0.154 | [+0.073, +0.252] | **yes** |
| Unpaired | +0.150 | [−0.043, +0.340] | no |

With 23 India-healthy patients, the unpaired comparison cannot detect a realistic intervention effect.
The paired one can.
