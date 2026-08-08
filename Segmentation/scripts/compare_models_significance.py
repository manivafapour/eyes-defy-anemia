"""
Compares 2+ trained segmentation models via their per-patient TEST-set
metrics (Segmentation/outputs/logs/{model_name}_test_per_patient.csv,
produced automatically by trainer_engine.py's run_study() -- see
segmentation_metrics.evaluate_final_test_set).

Answers two things a bare mean-Dice leaderboard cannot, both explicitly
requested by the project author for thesis-defense purposes:

1. Country-stratified breakdown -- does a model's Dice/IoU/etc. hold up
   equally for India and Italy patients, or does it silently degrade for
   one (this project has an established India/Italy demographic confound,
   CLAUDE.md Sec 0.5 -- the classification module's whole "defensibility
   programme" exists because of exactly this risk, and there's no reason
   segmentation would be immune to it).
2. Whether an apparent ranking between two models is actually
   statistically distinguishable on this project's small test sets (31
   palpebral / 33 forniceal_palpebral patients), or just a difference in
   point estimates that could easily be noise -- the same lesson this
   project's own classification module already learned the hard way
   (step1_cv_harness: India AUC differences across 12 models turned out to
   sit entirely inside the metric's own confidence-interval noise band,
   because the pair count backing it was too small to support a ranking
   claim). Both a paired Wilcoxon signed-rank test and a paired bootstrap
   95% CI are reported, since they can (and sometimes do) disagree, and
   this script's "significant" verdict requires both to agree, a
   deliberately conservative combined criterion.

Usage:
    python Segmentation/scripts/compare_models_significance.py MODEL_A MODEL_B [MODEL_C ...]
    python Segmentation/scripts/compare_models_significance.py MODEL_A MODEL_B --metric hd95

MODEL_NAME values are exactly the model_name strings used when that
model's train_pretrained/train_*.py script was run (e.g.
cnn_base_efficientnet_b1_unet_palpebral) -- i.e. the prefix on its
{model_name}_test_per_patient.csv file in outputs/logs/.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

METRIC_COLUMNS = ["dice", "iou", "precision", "recall", "hd95"]


def load_per_patient(model_name: str) -> pd.DataFrame:
    path = LOGS_DIR / f"{model_name}_test_per_patient.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- has {model_name} actually been trained via run_study() yet? "
            "This file is only written after a real training run completes, not by the "
            "structural/dry-run verification alone."
        )
    return pd.read_csv(path).set_index("patient_id")


def country_stratified_summary(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Per-country mean of every metric, plus an explicit 'overall' row --
    NOT just the pooled mean silently standing in for both groups."""
    summary = df.groupby("country")[METRIC_COLUMNS].mean(numeric_only=True)
    summary.loc["overall"] = df[METRIC_COLUMNS].mean(numeric_only=True)

    n_per_country = df.groupby("country").size()
    n_per_country["overall"] = len(df)
    summary.insert(0, "n_patients", n_per_country)
    summary.insert(0, "model_name", model_name)
    return summary


def paired_comparison(
    name_a: str, df_a: pd.DataFrame, name_b: str, df_b: pd.DataFrame, metric: str, n_bootstrap: int = 10000, seed: int = 42
) -> dict:
    """Paired Wilcoxon signed-rank test + a paired bootstrap 95% CI for the
    mean difference, computed over patients present in BOTH dataframes (an
    explicit inner join, not an assumption that both models were evaluated
    against identical patient sets -- they should be, since both come from
    the same tissue_type's test split, but this stays correct even if not).

    "Paired" matters here: the same 31-33 patients are shared across every
    model's test set, so a per-patient paired test (each patient is its own
    control) has far more statistical power than an unpaired comparison of
    two independent samples would -- exactly the right test for this
    situation, not a generic default."""
    common = df_a.index.intersection(df_b.index)
    if len(common) < 2:
        raise ValueError(f"Only {len(common)} common patients between {name_a} and {name_b} -- can't test.")

    a = df_a.loc[common, metric].to_numpy()
    b = df_b.loc[common, metric].to_numpy()

    valid = ~(np.isnan(a) | np.isnan(b))
    n_dropped = int((~valid).sum())
    a, b = a[valid], b[valid]
    diff = a - b

    if np.allclose(diff, 0):
        wilcoxon_p = 1.0
    else:
        _, wilcoxon_p = stats.wilcoxon(a, b)

    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_means = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    ci_excludes_zero = not (ci_low < 0 < ci_high)
    significant = bool(wilcoxon_p < 0.05 and ci_excludes_zero)

    return {
        "model_a": name_a,
        "model_b": name_b,
        "metric": metric,
        "n_patients": n,
        "n_dropped_nan": n_dropped,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff_a_minus_b": float(np.mean(diff)),
        "bootstrap_95ci_low": float(ci_low),
        "bootstrap_95ci_high": float(ci_high),
        "wilcoxon_p_value": float(wilcoxon_p),
        "significant_at_0.05": significant,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_names", nargs="+", help="model_name values as used by run_study(), 2 or more")
    parser.add_argument(
        "--metric", default="dice", choices=METRIC_COLUMNS,
        help="which metric to run the paired significance test on (default: dice)",
    )
    args = parser.parse_args()

    if len(args.model_names) < 2:
        print("Need at least 2 model_names to compare.", file=sys.stderr)
        sys.exit(1)

    dfs = {name: load_per_patient(name) for name in args.model_names}

    print("=== Country-stratified summary (per model) ===\n")
    summary_rows = []
    for name, df in dfs.items():
        table = country_stratified_summary(name, df)
        print(table.round(4).to_string())
        print()
        summary_rows.append(table)
    pd.concat(summary_rows).to_csv(LOGS_DIR / "model_comparison_country_stratified.csv")

    print(f"=== Paired significance tests (metric={args.metric}) ===\n")
    rows = [
        paired_comparison(name_a, dfs[name_a], name_b, dfs[name_b], metric=args.metric)
        for name_a, name_b in combinations(dfs, 2)
    ]
    comparison_df = pd.DataFrame(rows)
    print(comparison_df.round(4).to_string(index=False))

    out_path = LOGS_DIR / "model_comparison_significance.csv"
    comparison_df.to_csv(out_path, index=False)
    print(f"\nSaved country-stratified summary to {LOGS_DIR / 'model_comparison_country_stratified.csv'}")
    print(f"Saved significance comparison to {out_path}")


if __name__ == "__main__":
    main()
