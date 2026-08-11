# Step 1 Baseline -- Pooled Out-of-Fold Repeated Cross-Validation

Generated: 2026-08-08T13:45:29.711825+00:00

Frozen reference for Steps 2-5. Every later intervention must be compared against this
using a **paired** bootstrap on the difference (`cv_stats.paired_delta_auc`) with the
identical fold assignments recorded in each combo's `fold_manifest.json` -- not by
checking whether two independent confidence intervals overlap.

## Configuration

- Design: 5-fold x 5 repeats, stratified on country x label
- Seed: 42
- Bootstrap replicates: 2000
- Pool: train + val only; the 33-patient test split is sealed for Step 6
- Hyperparameters: each combo's own winning Optuna trial, reused verbatim (no re-tuning)

## Precision achieved

| | India pairs | Italy pairs |
|---|---|---|
| Single 70/15/15 split | 40 | 60 |
| Pooled out-of-fold | 1311 | 1680 |

## Results (sorted by India AUC)

| Combo | India AUC [95% CI] | Italy AUC [95% CI] | Overall AUC | Gap [95% CI] | Gap excl. 0 |
|---|---|---|---|---|---|
| convnext_tiny_palpebral | 0.722 [0.585, 0.840] | 0.842 [0.720, 0.937] | 0.858 | -0.121 [-0.294, +0.059] | no |
| convnext_tiny_forniceal_palpebral | 0.696 [0.562, 0.822] | 0.867 [0.755, 0.957] | 0.829 | -0.171 [-0.346, +0.005] | no |
| efficientnet_b0_forniceal_palpebral | 0.635 [0.465, 0.793] | 0.866 [0.747, 0.963] | 0.868 | -0.231 [-0.408, -0.054] | yes |
| resnet18_forniceal_palpebral | 0.625 [0.481, 0.758] | 0.896 [0.777, 0.971] | 0.804 | -0.271 [-0.441, -0.092] | yes |
| resnet18_palpebral | 0.593 [0.448, 0.729] | 0.850 [0.729, 0.945] | 0.828 | -0.257 [-0.429, -0.079] | yes |
| regnet_y_400mf_palpebral | 0.582 [0.431, 0.732] | 0.901 [0.788, 0.976] | 0.866 | -0.319 [-0.491, -0.127] | yes |
| densenet121_palpebral | 0.582 [0.426, 0.725] | 0.808 [0.667, 0.928] | 0.805 | -0.226 [-0.422, -0.024] | yes |
| mobilenet_v3_small_forniceal_palpebral | 0.581 [0.431, 0.726] | 0.732 [0.586, 0.857] | 0.738 | -0.151 [-0.340, +0.070] | no |
| densenet121_forniceal_palpebral | 0.561 [0.415, 0.695] | 0.862 [0.732, 0.963] | 0.808 | -0.301 [-0.498, -0.110] | yes |
| mobilenet_v3_small_palpebral | 0.518 [0.336, 0.664] | 0.569 [0.410, 0.730] | 0.582 | -0.051 [-0.290, +0.183] | no |
| regnet_y_400mf_forniceal_palpebral | 0.490 [0.330, 0.644] | 0.922 [0.845, 0.973] | 0.845 | -0.431 [-0.609, -0.249] | yes |
| efficientnet_b0_palpebral | 0.486 [0.336, 0.625] | 0.606 [0.437, 0.751] | 0.600 | -0.120 [-0.317, +0.087] | no |

## Gate results

### 5_negative_control -- PASS

```json
{
  "n_controls_run": 2,
  "results": [
    {
      "combo": "mobilenet_v3_small_palpebral_v2_clean",
      "mode": "global",
      "india_auc": 0.5203007518796993,
      "india_ci": [
        0.3659147869674185,
        0.6654135338345863
      ],
      "italy_auc": 0.51069033530572,
      "italy_ci": [
        0.3487179487179487,
        0.684437869822485
      ],
      "overall_auc": 0.5199174657118582,
      "overall_ci": [
        0.41982946959582473,
        0.6179147954848889
      ],
      "per_country_at_chance": true,
      "overall_at_chance": true,
      "passed": true
    },
    {
      "combo": "mobilenet_v3_small_palpebral_v2_clean",
      "mode": "within_country",
      "india_auc": 0.47383676582761247,
      "india_ci": [
        0.28832951945080093,
        0.6438215102974827
      ],
      "italy_auc": 0.47059523809523807,
      "italy_ci": [
        0.32617559523809525,
        0.6261904761904762
      ],
      "overall_auc": 0.5649472023303799,
      "overall_ci": [
        0.4697141643403326,
        0.6612847432940888
      ],
      "per_country_at_chance": true,
      "overall_above_chance_as_expected": false,
      "passed": true
    }
  ],
  "passed": true,
  "note": "Per-country AUC must cover 0.50 once labels are destroyed; otherwise the harness leaks."
}
```

### 7_plausibility -- PASS

```json
{
  "expected_range": [
    0.4,
    0.9
  ],
  "observed_min": 0.48634630053394357,
  "observed_max": 0.7218916857360793,
  "n_outside": 0,
  "combos_outside": [],
  "passed": true,
  "note": "Outside this range => suspect the harness before believing the result."
}
```

### 8_precision -- FAIL

```json
{
  "threshold_ci_half_width": 0.12,
  "worst_observed": 0.16401601830663606,
  "median_observed": 0.14612414187643002,
  "n_failing": 12,
  "failing_combos": [
    "convnext_tiny_forniceal_palpebral_v2_clean",
    "convnext_tiny_palpebral_v2_clean",
    "densenet121_forniceal_palpebral_v2_clean",
    "densenet121_palpebral_v2_clean",
    "efficientnet_b0_forniceal_palpebral_v2_clean",
    "efficientnet_b0_palpebral_v2_clean",
    "mobilenet_v3_small_forniceal_palpebral_v2_clean",
    "mobilenet_v3_small_palpebral_v2_clean",
    "regnet_y_400mf_forniceal_palpebral_v2_clean",
    "regnet_y_400mf_palpebral_v2_clean",
    "resnet18_forniceal_palpebral_v2_clean",
    "resnet18_palpebral_v2_clean"
  ],
  "passed": false
}
```

## Verdict

**Step 1 NOT CLEAR.** One or more gates failed -- resolve before starting Step 2. Running interventions against an unverified harness would produce unfalsifiable results.
