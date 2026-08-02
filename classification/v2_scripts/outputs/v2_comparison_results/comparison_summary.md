# Phase 4 v2 Classification — 18-Combo Comparison Summary

**Generated:** 2026-07-31
**Scope:** all 9 architectures × 2 tissue types (18 combos total), unified v2 protocol (100-epoch ceiling, `dropout_rate` tuned, 12-trial Optuna search each). Batch 1 (14 light/medium combos) + Batch 2 (ViT-B/16, ViT-L/16 × 2 tissue types) combined for the first time in this analysis, per the deliberate pause recorded in `.project_memory/02_current_status.md`.

Full numeric detail: [`comparison_table.csv`](comparison_table.csv) / [`comparison_table.md`](comparison_table.md). Plots: `f1_comparison.png`, `metrics_comparison.png`, `auc_gap_comparison.png`.

---

## Absolute best model: **EfficientNet-B0 / forniceal_palpebral**

| Metric | Value |
|---|---|
| Overall F1 | **0.933** |
| Overall Balanced Accuracy | **0.941** |
| Overall AUC | 0.887 |
| Overall Sensitivity (recall on anemic class) | **1.000** |
| Overall Specificity | 0.882 |
| India AUC | 0.750 |
| Italy AUC | 0.942 |
| India/Italy AUC gap | 0.192 (rank 6th-smallest of 18 — mid-pack, not an outlier) |

This is the clear winner on F1, the metric this project has used as its headline ranking criterion since the original 6-combo comparison, and it also leads on Balanced Accuracy — the two metrics agree, which was explicitly the standard the original analysis (`CLAUDE.md`-equivalent, `02_current_status.md` "Kaggle results" section) used to call a result unambiguous rather than a close call between metrics. Its India/Italy AUC gap (0.192) is not the smallest in the roster, but it sits comfortably in the better half (6th of 18) rather than among the confound-exploiting outliers — it is not "winning by ignoring India," unlike some other high-F1 combos below.

**This confirms, on the full 18-architecture roster (including both ~87M- and ~304M-parameter transformers), the same combo already identified as the best batch-1 performer and independently as the strongest single candidate for the dedicated 5-fold CV deep-dive** (`classification/scripts/efficientnet_b0_forniceal_5fold_cv/`, in progress on Kaggle as of this analysis). A 5.3M-parameter CNN outperforming ViT-L/16 (304M parameters) by every headline metric is consistent with this project's standing design rationale for frozen-backbone transfer learning: with only ~151 training patients, a larger backbone has more capacity to overfit the classification head's limited training signal, not more useful capacity to exploit.

## Runner-up worth naming: best confound-handling

**MobileNetV3-Small / palpebral** — F1 0.839 (rank 8), AUC 0.857, India/Italy AUC gap **0.017**, essentially no gap at all (the smallest of all 18 combos by a wide margin — the next-closest is Swin-Tiny/palpebral at 0.050). If the thesis prioritizes demonstrated confound-robustness over raw performance, this is the strongest evidence-backed alternative, continuing the same "best-overall vs. best-confound-handling" framing already used for the original 6-combo and previewed for batch-1 results.

The user's instruction was to select a single absolute best model — that is EfficientNet-B0/forniceal_palpebral above — but the two disagreeing is itself a real, reportable finding rather than something to paper over, consistent with how this project has always presented these results.

## New finding: closes a pending roadmap item

`01_roadmap.md` explicitly flagged, as a not-yet-answered question: *"whether global-attention transformers (ViT-B/16, ViT-L/16) show a larger confound gap than local-attention (Swin-Tiny) or no-attention (the 6 CNNs) architectures."* Now answerable with all 18 combos in hand:

| Architecture group | Mean India/Italy AUC gap | n |
|---|---|---|
| No-attention CNNs (6 architectures) | **0.270** | 12 |
| Local-attention transformer (Swin-Tiny) | 0.215 | 2 |
| Global-attention transformers (ViT-B/16, ViT-L/16) | 0.211 | 4 |

**The hypothesis is not supported — if anything, the opposite pattern holds.** Both transformer families show a *smaller* average confound gap than the CNN group, not a larger one. Global- and local-attention transformers land within 0.004 of each other (essentially tied), while CNNs average noticeably higher. This is based on only 6 transformer combos vs. 12 CNN combos, and per-combo variance is large (CNN range alone is 0.017–0.567), so this should be read as a suggestive pattern worth a footnote, not a statistically settled claim — but it directly contradicts the pre-registered hypothesis and is worth reporting as such rather than silently dropping the question now that it's answerable.

## Secondary finding: tissue type trade-off

| Tissue type | Mean F1 | Mean India/Italy AUC gap |
|---|---|---|
| palpebral | **0.859** | **0.206** |
| forniceal_palpebral | 0.821 | 0.297 |

Averaged across all 9 architectures, **palpebral is the more reliable tissue type on both axes** — higher mean F1 *and* a smaller mean confound gap. This nuances the top-line result: the single best individual model uses forniceal_palpebral, but that is the exception rather than the rule across the roster — most architectures do better, and more evenly across countries, on the plain palpebral crop. Consistent with the original 6-combo finding that only one of three architectures (EfficientNet-B0) benefited from forniceal_palpebral over palpebral at all.

## Notable individual outlier

**ResNet18 / forniceal_palpebral** — last place by F1 (0.765) *and* the single worst confound gap in the entire roster (0.567, India AUC 0.375 — sub-chance). This matches the same architecture/tissue combination already flagged as "the clearest evidence of confound exploitation" in the original 6-combo analysis (`02_current_status.md`), now confirmed to remain the worst offender even after being retrained under the improved v2 protocol (dropout tuning, 100-epoch ceiling) and compared against 12 additional architectures.

---

*Metrics are validation-set performance from each combo's best Optuna trial (of 12), not test-set performance — consistent with how every result in this module has been reported to date (test set is held out project-wide, not yet touched by any Phase 4 classification run).*
