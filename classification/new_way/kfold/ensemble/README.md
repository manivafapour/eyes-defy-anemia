# `new_way/kfold/ensemble/` — post-hoc ensembles over the 18 K-fold checkpoints

All techniques here combine the **18 frozen-backbone checkpoints** produced by
`new_way/kfold/` (6 architectures × 3 folds, `Output/version2/checkpoints/`,
trained on the offline country-balanced data). Every number below is on the
**sealed 33-patient test split** (15 India + 18 Italy), which no K-fold split
ever touches (`build_folds()` only partitions the pooled train+val patients).

## Files

| File | What it does |
|---|---|
| `ensemble_engine.py` | Inference + merging logic: per-checkpoint prediction (cached), soft-voting, uniform model soups, greedy soup, OOF stacking, plus the built-in correctness checks (`verify_against_recorded`, `verify_oof_fold_sizes`, `verify_soup_degenerates_to_checkpoint`). |
| `run_ensemble.py` | Entry point — runs the soft-voting / soup / greedy-soup / stacking comparison, writes `Output/version2/logs/ensemble_comparison.{csv,json}` + `plots/ensemble_comparison.png`. |
| `bagging_boosting.py` | Bagging (B bootstrap-resampled heads, early-stopped on out-of-bag patients) and boosting (AdaBoost.M1 / SAMME over the head). These two **train new heads** — the only techniques here that do — but it is cheap because only the `Dropout→Linear` head updates. |
| `run_bagging_boosting.py` | Entry point for bagging + boosting (currently scoped to `regnet_y_16gf` and `maxvit_t`), writes `Output/version2/logs/bagging_boosting_cache/`. |
| `run_final_comparison.py` | Merges every technique's saved results into `Output/version2/logs/all_ensemble_techniques_comparison.csv` + `plots/all_ensemble_techniques_comparison.png`, with a leak-free / OPTIMISTIC label per row. Needs the two runners above to have run first. |
| `ensemble_all_results.xlsx` | Spreadsheet mirror of the final comparison. |

## How to run

```bash
python classification/new_way/kfold/ensemble/run_ensemble.py
python classification/new_way/kfold/ensemble/run_bagging_boosting.py
python classification/new_way/kfold/ensemble/run_final_comparison.py
```

Requires `Output/version2/checkpoints/` (the 18 fold checkpoints + the
`bagging_boosting/` sub-folder) on disk — `.gitignore`'d, not in the repo.

## Results (sealed test set, sorted by F1)

| Technique | Best config | F1 | Acc | AUC | Leakage |
|---|---|---|---|---|---|
| **Bagging** | `maxvit_t_bagging` (5 heads) | **0.929** | **0.939** | **0.981** | leak-free |
| Greedy soup | `maxvit_t_greedy_soup` | 0.897 | 0.909 | 0.962 | **OPTIMISTIC** (subset chosen on the test set) |
| Soft-voting | `top2_best_fold` (MaxViT-T + RegNetY-16GF) | 0.867 | 0.879 | 0.940 | leak-free |
| Stacking (OOF) | `stack_top2` / `stack_top3` | 0.828 | 0.848 | 0.940 | leak-free |
| Uniform soup | `maxvit_t_soup_all_folds` | 0.828 | 0.848 | 0.955 | leak-free |
| Boosting | `maxvit_t_boosting` | 0.750 | 0.818 | 0.917 | leak-free |

**Headline (leak-free): MaxViT-Tiny bagging — F1 0.929 / Acc 0.939 / AUC 0.981**,
confusion matrix `[[17,2],[0,14]]`, India AUC 0.841 / Italy AUC 1.000.

## Caveats

- **33-patient test set.** Every metric has a wide confidence interval; India
  AUC rests on ~15 patients. Report the direction, not a fine ranking.
- **Greedy soup is selection-leaky.** This project's 3-fold split has no
  leak-free multi-fold validation set, so its keep/discard decisions are made
  against the same test set it is then scored on. Reported as
  `*_greedy_soup_TEST_SELECTED` and kept in its own "kind" for that reason —
  read it as "best subset this test set happened to reward", not an unbiased
  estimate.
- **Boosting underperformed, as predicted.** Classic boosting assumes weak
  learners; a frozen ImageNet backbone + linear head is not weak, so
  reweighting adds little signal round-to-round. `regnet_y_16gf_boosting`
  collapsed to F1 0.50. Report as an honest negative or omit.
- **Mixing architectures did not help.** `top2` beat `top3` / `top5` / `all6`;
  adding weaker members (especially EfficientNet-B3, which also uses the
  `forniceal_palpebral` crop) diluted the vote.
- **Bagging/boosting ran for 2 architectures only** (`regnet_y_16gf`,
  `maxvit_t`). Extending to the rest is a natural next step.
