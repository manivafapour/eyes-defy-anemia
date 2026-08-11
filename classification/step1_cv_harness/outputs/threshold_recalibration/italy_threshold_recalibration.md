# Step 1 -- Italy Threshold Recalibration

Generated: 2026-08-09T08:00:15.323609+00:00

Nested leave-one-fold-out threshold selection (grid search, F1-maximizing, 0.01-0.99 step 0.01) applied to Italy's out-of-fold predictions from the pooled CV baseline. Every fold's reported metric uses a threshold selected from the OTHER 4 folds in its repeat only -- never from itself -- so these numbers are a legitimate estimate of a pre-committed threshold's performance on unseen Italy patients, not an optimistic in-sample best case.

India is left at the fixed 0.5 threshold throughout: India's deficit is AUC (discrimination), not calibration, so recalibrating its threshold would not address the actual problem -- see `outputs/baseline/step1_baseline.md`.

## Results, ranked by F1 improvement (descending)

| Combo | Italy F1 @0.5 | Italy F1 recal. | ΔF1 | ΔPrecision | ΔRecall | Threshold (mean±SD) | Overall F1 @0.5 | Overall F1 mixed policy |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_small_palpebral | 0.301 | 0.355 | +0.054 | +0.079 | +0.140 | 0.44±0.33 | 0.481 | 0.499 |
| vit_l_16_palpebral | 0.678 | 0.720 | +0.041 | +0.146 | -0.080 | 0.75±0.08 | 0.765 | 0.780 |
| resnet18_forniceal_palpebral | 0.607 | 0.629 | +0.022 | +0.130 | -0.158 | 0.58±0.04 | 0.668 | 0.680 |
| densenet121_palpebral | 0.555 | 0.572 | +0.017 | +0.140 | -0.160 | 0.55±0.06 | 0.716 | 0.739 |
| efficientnet_b0_forniceal_palpebral | 0.629 | 0.636 | +0.007 | -0.035 | +0.063 | 0.45±0.09 | 0.796 | 0.794 |
| resnet18_palpebral | 0.678 | 0.681 | +0.003 | +0.063 | -0.070 | 0.60±0.09 | 0.742 | 0.744 |
| densenet121_forniceal_palpebral | 0.632 | 0.631 | -0.001 | +0.072 | -0.116 | 0.60±0.08 | 0.692 | 0.693 |
| convnext_tiny_forniceal_palpebral | 0.612 | 0.603 | -0.009 | +0.051 | -0.126 | 0.74±0.24 | 0.717 | 0.721 |
| efficientnet_b0_palpebral | 0.324 | 0.307 | -0.017 | -0.055 | +0.080 | 0.42±0.14 | 0.478 | 0.462 |
| regnet_y_400mf_forniceal_palpebral | 0.628 | 0.610 | -0.018 | -0.171 | +0.116 | 0.42±0.06 | 0.750 | 0.740 |
| swin_t_forniceal_palpebral | 0.611 | 0.591 | -0.020 | +0.072 | -0.147 | 0.70±0.25 | 0.702 | 0.701 |
| convnext_tiny_palpebral | 0.637 | 0.611 | -0.026 | +0.015 | -0.080 | 0.52±0.07 | 0.762 | 0.757 |
| vit_b_16_forniceal_palpebral | 0.581 | 0.552 | -0.030 | -0.045 | -0.011 | 0.49±0.10 | 0.737 | 0.727 |
| vit_b_16_palpebral | 0.606 | 0.575 | -0.031 | +0.014 | -0.080 | 0.53±0.08 | 0.773 | 0.768 |
| swin_t_palpebral | 0.593 | 0.556 | -0.036 | +0.013 | -0.090 | 0.53±0.17 | 0.724 | 0.714 |
| mobilenet_v3_small_forniceal_palpebral | 0.475 | 0.438 | -0.038 | +0.007 | -0.168 | 0.56±0.05 | 0.677 | 0.677 |
| vit_l_16_forniceal_palpebral | 0.695 | 0.638 | -0.057 | -0.059 | -0.042 | 0.49±0.10 | 0.757 | 0.741 |
| regnet_y_400mf_palpebral | 0.678 | 0.602 | -0.075 | -0.236 | +0.070 | 0.40±0.09 | 0.769 | 0.744 |

## Summary

- 6/18 combos improved Italy F1 under honest nested recalibration.
- Median delta-F1 (Italy, nested): -0.017; mean: -0.012
- Median recalibrated threshold (Italy): 0.53 (baseline was fixed at 0.5)

## Naive (leaky) vs. honest (nested) -- documented negative control

Selecting a single F1-maximizing threshold from ALL of a combo's Italy out-of-fold predictions and then evaluating it on that SAME data (no nesting -- exactly the mistake this module's design exists to avoid) makes it LOOK like recalibration helps in **18/18** combos. Once evaluated honestly with leave-one-fold-out nesting above, only **6/18** actually improve, and the median effect is slightly negative (-0.017). This gap is measured directly on this project's own data, not asserted from general principle -- see `naive_leaky_italy_f1` in the CSV/JSON for the per-combo naive numbers. Conclusion: most of the apparent gain from a fixed post-hoc threshold shift on Italy is overfitting to which ~104 Italy patients happened to be in the pool, not a real, generalizable improvement -- consistent with the small per-fold sample size (~20 Italy patients held out per fold).

## Deployment threshold vs. reported metric -- do not conflate these

The `italy_deployment_threshold` column (CSV/JSON only, not shown above) is selected once from ALL of a combo's Italy out-of-fold predictions pooled together, with no nesting. It is what you would actually configure a deployed model with -- but it is NOT a valid basis for the F1 numbers reported here, since it is chosen on the same data it would then be tested on. The nested procedure above intentionally produces up to 25 different per-fold thresholds rather than one, because that is what an honest leave-one-out estimate requires; the single deployment threshold is a separate, practical output with a different (weaker) evidentiary status.
