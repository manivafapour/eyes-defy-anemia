# Step 1 Baseline -- Pooled Out-of-Fold Repeated Cross-Validation

Generated: 2026-08-08T20:26:51.432932+00:00

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
| vit_l_16_forniceal_palpebral | 0.699 [0.562, 0.821] | 0.912 [0.835, 0.970] | 0.861 | -0.214 [-0.362, -0.062] | yes |
| vit_l_16_palpebral | 0.683 [0.538, 0.812] | 0.857 [0.730, 0.958] | 0.854 | -0.174 [-0.355, +0.022] | no |
| vit_b_16_forniceal_palpebral | 0.680 [0.529, 0.809] | 0.842 [0.711, 0.941] | 0.841 | -0.162 [-0.347, +0.027] | no |
| swin_t_forniceal_palpebral | 0.648 [0.497, 0.783] | 0.883 [0.766, 0.959] | 0.809 | -0.235 [-0.402, -0.072] | yes |
| swin_t_palpebral | 0.634 [0.486, 0.777] | 0.820 [0.676, 0.939] | 0.825 | -0.187 [-0.397, +0.034] | no |

## Gate results

### 5_negative_control -- FAIL

```json
{
  "n_controls_run": 0,
  "results": [],
  "passed": false,
  "note": "Not run -- Step 1 cannot be cleared without at least one label-shuffle control."
}
```

### 7_plausibility -- PASS

```json
{
  "expected_range": [
    0.4,
    0.9
  ],
  "observed_min": 0.633714721586575,
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
  "worst_observed": 0.14530892448512583,
  "median_observed": 0.13864893211289092,
  "n_failing": 6,
  "failing_combos": [
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
