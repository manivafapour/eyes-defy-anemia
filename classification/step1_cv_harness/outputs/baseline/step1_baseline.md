# Step 1 Baseline -- Pooled Out-of-Fold Repeated Cross-Validation

Generated: 2026-08-08T21:51:16.752883+00:00

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
| vit_b_16_palpebral | 0.740 [0.605, 0.855] | 0.846 [0.722, 0.941] | 0.874 | -0.106 [-0.282, +0.069] | no |
| convnext_tiny_palpebral | 0.719 [0.584, 0.840] | 0.843 [0.718, 0.937] | 0.858 | -0.124 [-0.296, +0.051] | no |
| vit_l_16_forniceal_palpebral | 0.699 [0.562, 0.821] | 0.912 [0.835, 0.970] | 0.861 | -0.214 [-0.362, -0.062] | yes |
| convnext_tiny_forniceal_palpebral | 0.697 [0.562, 0.831] | 0.871 [0.758, 0.959] | 0.831 | -0.174 [-0.345, -0.010] | yes |
| vit_l_16_palpebral | 0.683 [0.538, 0.812] | 0.857 [0.730, 0.958] | 0.854 | -0.174 [-0.355, +0.022] | no |
| vit_b_16_forniceal_palpebral | 0.680 [0.529, 0.809] | 0.842 [0.711, 0.941] | 0.841 | -0.162 [-0.347, +0.027] | no |
| swin_t_forniceal_palpebral | 0.648 [0.497, 0.783] | 0.883 [0.766, 0.959] | 0.809 | -0.235 [-0.402, -0.072] | yes |
| swin_t_palpebral | 0.634 [0.486, 0.777] | 0.820 [0.676, 0.939] | 0.825 | -0.187 [-0.397, +0.034] | no |
| efficientnet_b0_forniceal_palpebral | 0.632 [0.474, 0.780] | 0.862 [0.741, 0.957] | 0.867 | -0.230 [-0.402, -0.051] | yes |
| resnet18_forniceal_palpebral | 0.625 [0.481, 0.765] | 0.894 [0.777, 0.969] | 0.803 | -0.269 [-0.434, -0.096] | yes |
| resnet18_palpebral | 0.592 [0.447, 0.729] | 0.849 [0.723, 0.945] | 0.831 | -0.257 [-0.425, -0.086] | yes |
| mobilenet_v3_small_forniceal_palpebral | 0.588 [0.410, 0.751] | 0.753 [0.616, 0.872] | 0.737 | -0.166 [-0.389, +0.053] | no |
| densenet121_palpebral | 0.587 [0.423, 0.738] | 0.819 [0.685, 0.928] | 0.806 | -0.233 [-0.417, -0.052] | yes |
| densenet121_forniceal_palpebral | 0.581 [0.432, 0.719] | 0.854 [0.729, 0.954] | 0.808 | -0.272 [-0.468, -0.074] | yes |
| regnet_y_400mf_palpebral | 0.575 [0.425, 0.725] | 0.902 [0.801, 0.973] | 0.866 | -0.327 [-0.494, -0.150] | yes |
| mobilenet_v3_small_palpebral | 0.522 [0.373, 0.669] | 0.604 [0.434, 0.768] | 0.579 | -0.082 [-0.313, +0.143] | no |
| efficientnet_b0_palpebral | 0.496 [0.320, 0.681] | 0.660 [0.487, 0.805] | 0.654 | -0.164 [-0.391, +0.052] | no |
| regnet_y_400mf_forniceal_palpebral | 0.490 [0.329, 0.643] | 0.920 [0.847, 0.970] | 0.844 | -0.430 [-0.604, -0.246] | yes |

## Gate results

### 5_negative_control -- PASS

```json
{
  "n_controls_run": 2,
  "results": [
    {
      "combo": "mobilenet_v3_small_palpebral_v2_clean",
      "mode": "global",
      "india_auc": 0.4989974937343358,
      "india_ci": [
        0.3696428571428571,
        0.6278352130325814
      ],
      "italy_auc": 0.4939644970414202,
      "italy_ci": [
        0.3620907297830375,
        0.6272189349112426
      ],
      "overall_auc": 0.503119310595946,
      "overall_ci": [
        0.41218291054739653,
        0.5918376016506857
      ],
      "per_country_at_chance": true,
      "overall_at_chance": true,
      "passed": true
    },
    {
      "combo": "mobilenet_v3_small_palpebral_v2_clean",
      "mode": "within_country",
      "india_auc": 0.4698703279938978,
      "india_ci": [
        0.2959572845156369,
        0.6362128146453085
      ],
      "italy_auc": 0.4769047619047619,
      "italy_ci": [
        0.2976041666666666,
        0.6518005952380952
      ],
      "overall_auc": 0.5522514868309262,
      "overall_ci": [
        0.4615790751304771,
        0.6390460007282436
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
  "observed_min": 0.4900076277650648,
  "observed_max": 0.7398932112890924,
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
  "worst_observed": 0.1807780320366133,
  "median_observed": 0.1430444317315026,
  "n_failing": 18,
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
    "resnet18_palpebral_v2_clean",
    "swin_t_forniceal_palpebral_v2_clean",
    "swin_t_palpebral_v2_clean",
    "vit_b_16_forniceal_palpebral_v2_clean",
    "vit_b_16_palpebral_v2_clean",
    "vit_l_16_forniceal_palpebral_v2_clean",
    "vit_l_16_palpebral_v2_clean"
  ],
  "passed": false
}
```

## Verdict

**Step 1 NOT CLEAR.** One or more gates failed -- resolve before starting Step 2. Running interventions against an unverified harness would produce unfalsifiable results.
