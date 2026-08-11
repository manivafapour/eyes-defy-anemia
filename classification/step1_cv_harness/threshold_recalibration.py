"""
Step 1 -- Measurement Harness: per-country threshold recalibration (Italy).

CONTEXT. Gate 8 evaluates AUC precision only; it says nothing about
precision/recall/F1 at the conventional 0.5 threshold. A fresh, country-
stratified computation of those metrics (from oof_predictions.csv, pooled
within each repeat then averaged across the 5 repeats -- same design
cv_stats.bootstrap_auc_cis() uses for AUC) showed Italy's F1 sitting well
below India's across all 18 combos, despite Italy's AUC being consistently
*higher* than India's. The mechanism is base-rate sensitivity: Italy's pool
is ~19% anemic, so a fixed 0.5 cutoff -- tuned for nothing in particular --
sits in the wrong place for how rare positives actually are. AUC is
threshold-independent and does not see this; F1 at a fixed threshold does.

WHAT THIS MODULE DOES. Recomputes Italy's operating threshold instead of
its model. For each repeat, each of the 5 outer folds' Italy predictions
are evaluated using a threshold selected from the OTHER 4 folds in that
repeat only (grid search over 0.01-0.99 maximizing F1) -- so no fold ever
sets its own operating point, which is what makes the resulting
precision/recall/F1 a legitimate estimate of what a pre-committed threshold
would achieve on unseen patients, not an optimistic best-case that leaks the
evaluation data into the threshold choice. This is nested leave-one-fold-out
threshold selection, and it costs nothing beyond re-reading each combo's
already-computed oof_predictions.csv -- no retraining, no GPU.

Two threshold values are reported and must not be conflated:
  - The NESTED recalibrated metric (precision/recall/F1) is what would
    actually be observed if a fold's threshold is always chosen without
    seeing that fold -- this is the number to report/compare against the
    baseline.
  - The DEPLOYMENT threshold is selected once from ALL of a combo's Italy
    out-of-fold predictions pooled together (no nesting). It is NOT a valid
    basis for the reported metric above (selected on the same data it would
    then be "tested" on) -- it exists only because a deployed system needs a
    single number to configure, and the nested procedure legitimately
    produces 25 different per-fold thresholds rather than one.

Usage:
    python classification/step1_cv_harness/threshold_recalibration.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_config import OUTPUTS_DIR  # noqa: E402

RECAL_DIR = OUTPUTS_DIR / "threshold_recalibration"
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
FIXED_BASELINE_THRESHOLD = 0.5


def _discover_real_combos(outputs_dir: Path) -> list:
    """Same dynamic-discovery + shuffle-control-exclusion convention as
    aggregate_baseline.py's _load_metrics -- combos are found by globbing,
    not hardcoded, and the label-shuffle negative controls are skipped since
    they have no real threshold to recalibrate."""
    names = []
    for metrics_path in sorted(outputs_dir.glob("*/cv_metrics.json")):
        with open(metrics_path) as f:
            m = json.load(f)
        if m.get("shuffle_control", "none") == "none":
            names.append(metrics_path.parent.name)
    return names


def _counts(labels: np.ndarray, preds: np.ndarray) -> tuple:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tp), int(fp), int(fn), int(tn)


def _prf_from_counts(tp: int, fp: int, fn: int) -> tuple:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def best_threshold_for_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    """Grid search over THRESHOLD_GRID maximizing F1. Ties are broken by the
    MEDIAN of every threshold tied for the best F1 -- with ~80-84 patients in
    a selection set, several adjacent 0.01 steps commonly tie exactly, and
    taking the first or the extreme of that tied range would claim a
    precision the grid search does not actually have."""
    best_f1 = -1.0
    tied = []
    for t in THRESHOLD_GRID:
        preds = (probs > t).astype(float)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            tied = [t]
        elif abs(f1 - best_f1) <= 1e-12:
            tied.append(t)
    return float(np.median(tied))


def _country_slice(df: pd.DataFrame, country: str) -> pd.DataFrame:
    return df if country == "Overall" else df[df["country"] == country]


def baseline_at_fixed_threshold(df: pd.DataFrame, country: str, threshold: float = FIXED_BASELINE_THRESHOLD) -> dict:
    """Pool within each repeat, average across repeats -- no nesting needed
    since nothing is selected, the threshold is fixed."""
    sub = _country_slice(df, country)
    per_repeat = {"precision": [], "recall": [], "f1": []}
    for r in sorted(sub["repeat"].unique()):
        rows = sub[sub["repeat"] == r]
        labels = rows["label"].to_numpy()
        preds = (rows["prob"].to_numpy() > threshold).astype(float)
        per_repeat["precision"].append(precision_score(labels, preds, zero_division=0))
        per_repeat["recall"].append(recall_score(labels, preds, zero_division=0))
        per_repeat["f1"].append(f1_score(labels, preds, zero_division=0))
    return {m: float(np.mean(v)) for m, v in per_repeat.items()}


def nested_recalibrate(df: pd.DataFrame, country: str) -> dict:
    """Leave-one-fold-out nested threshold selection. Within each repeat, the
    5 folds' held-out confusion counts are pooled (each fold's own threshold
    selected only from the other 4 folds in that repeat) into one
    repeat-level precision/recall/F1; the point estimate is the mean of the
    5 repeat-level values -- the same 'point estimate = mean of per-repeat
    pooled values' convention cv_stats.bootstrap_auc_cis() uses for AUC."""
    sub = _country_slice(df, country)
    per_repeat_prf = {"precision": [], "recall": [], "f1": []}
    selected_thresholds = []

    for r in sorted(sub["repeat"].unique()):
        rep_rows = sub[sub["repeat"] == r]
        tp_sum = fp_sum = fn_sum = 0
        for f in sorted(rep_rows["fold"].unique()):
            sel = rep_rows[rep_rows["fold"] != f]
            ev = rep_rows[rep_rows["fold"] == f]
            t = best_threshold_for_f1(sel["label"].to_numpy(), sel["prob"].to_numpy())
            selected_thresholds.append(t)
            preds = (ev["prob"].to_numpy() > t).astype(float)
            tp, fp, fn, _tn = _counts(ev["label"].to_numpy(), preds)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        precision, recall, f1 = _prf_from_counts(tp_sum, fp_sum, fn_sum)
        per_repeat_prf["precision"].append(precision)
        per_repeat_prf["recall"].append(recall)
        per_repeat_prf["f1"].append(f1)

    result = {m: float(np.mean(v)) for m, v in per_repeat_prf.items()}
    result["threshold_mean"] = float(np.mean(selected_thresholds))
    result["threshold_median"] = float(np.median(selected_thresholds))
    result["threshold_std"] = float(np.std(selected_thresholds, ddof=1))
    result["threshold_min"] = float(np.min(selected_thresholds))
    result["threshold_max"] = float(np.max(selected_thresholds))
    result["n_threshold_selections"] = len(selected_thresholds)
    result["all_selected_thresholds"] = selected_thresholds
    return result


def deployment_threshold(df: pd.DataFrame, country: str) -> float:
    """Non-nested: selected once from ALL out-of-fold predictions for this
    country, pooled across every repeat and fold. For configuring an actual
    deployed model only -- NOT a valid basis for the reported metric, since
    it is chosen on the same data it would then be measured against."""
    sub = _country_slice(df, country)
    return best_threshold_for_f1(sub["label"].to_numpy(), sub["prob"].to_numpy())


def naive_leaky_recalibrate(df: pd.DataFrame, country: str) -> dict:
    """Same threshold search as deployment_threshold, but then evaluated on
    the SAME data it was selected from -- the mistake this whole module
    exists to avoid. Computed and reported purely as a documented negative
    control: it demonstrates, on this project's own data, how much of the
    apparent improvement from threshold tuning is an artifact of not nesting
    the selection. Never present these numbers as the recalibration result."""
    sub = _country_slice(df, country)
    t = best_threshold_for_f1(sub["label"].to_numpy(), sub["prob"].to_numpy())
    preds = (sub["prob"].to_numpy() > t).astype(float)
    labels = sub["label"].to_numpy()
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": t,
    }


def mixed_overall_after_recalibration(df: pd.DataFrame) -> dict:
    """Overall precision/recall/F1 under a mixed policy: India keeps the
    fixed 0.5 threshold (India's problem is discrimination/AUC, not
    calibration -- recalibrating its threshold would not address the actual
    deficiency), Italy uses the nested leave-one-fold-out recalibrated
    threshold. Shows the practical effect on the whole cohort of fixing only
    the country that actually has a threshold problem."""
    per_repeat_prf = {"precision": [], "recall": [], "f1": []}
    for r in sorted(df["repeat"].unique()):
        rep_rows = df[df["repeat"] == r]
        tp_sum = fp_sum = fn_sum = 0

        india_rows = rep_rows[rep_rows["country"] == "India"]
        preds = (india_rows["prob"].to_numpy() > FIXED_BASELINE_THRESHOLD).astype(float)
        tp, fp, fn, _tn = _counts(india_rows["label"].to_numpy(), preds)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

        italy_rows = rep_rows[rep_rows["country"] == "Italy"]
        for f in sorted(italy_rows["fold"].unique()):
            sel = italy_rows[italy_rows["fold"] != f]
            ev = italy_rows[italy_rows["fold"] == f]
            t = best_threshold_for_f1(sel["label"].to_numpy(), sel["prob"].to_numpy())
            preds = (ev["prob"].to_numpy() > t).astype(float)
            tp, fp, fn, _tn = _counts(ev["label"].to_numpy(), preds)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

        precision, recall, f1 = _prf_from_counts(tp_sum, fp_sum, fn_sum)
        per_repeat_prf["precision"].append(precision)
        per_repeat_prf["recall"].append(recall)
        per_repeat_prf["f1"].append(f1)
    return {m: float(np.mean(v)) for m, v in per_repeat_prf.items()}


def process_combo(outputs_dir: Path, combo: str) -> dict:
    df = pd.read_csv(outputs_dir / combo / "oof_predictions.csv")

    baseline_italy = baseline_at_fixed_threshold(df, "Italy")
    recal_italy = nested_recalibrate(df, "Italy")
    baseline_overall = baseline_at_fixed_threshold(df, "Overall")
    mixed_overall = mixed_overall_after_recalibration(df)
    deploy_t_italy = deployment_threshold(df, "Italy")
    naive_italy = naive_leaky_recalibrate(df, "Italy")

    return {
        "combo": combo,
        "baseline_italy_precision": baseline_italy["precision"],
        "baseline_italy_recall": baseline_italy["recall"],
        "baseline_italy_f1": baseline_italy["f1"],
        "recal_italy_precision": recal_italy["precision"],
        "recal_italy_recall": recal_italy["recall"],
        "recal_italy_f1": recal_italy["f1"],
        "delta_italy_precision": recal_italy["precision"] - baseline_italy["precision"],
        "delta_italy_recall": recal_italy["recall"] - baseline_italy["recall"],
        "delta_italy_f1": recal_italy["f1"] - baseline_italy["f1"],
        "italy_threshold_mean": recal_italy["threshold_mean"],
        "italy_threshold_median": recal_italy["threshold_median"],
        "italy_threshold_std": recal_italy["threshold_std"],
        "italy_threshold_min": recal_italy["threshold_min"],
        "italy_threshold_max": recal_italy["threshold_max"],
        "italy_deployment_threshold": deploy_t_italy,
        "naive_leaky_italy_precision": naive_italy["precision"],
        "naive_leaky_italy_recall": naive_italy["recall"],
        "naive_leaky_italy_f1": naive_italy["f1"],
        "baseline_overall_precision": baseline_overall["precision"],
        "baseline_overall_recall": baseline_overall["recall"],
        "baseline_overall_f1": baseline_overall["f1"],
        "mixed_overall_precision": mixed_overall["precision"],
        "mixed_overall_recall": mixed_overall["recall"],
        "mixed_overall_f1": mixed_overall["f1"],
        "delta_mixed_overall_f1": mixed_overall["f1"] - baseline_overall["f1"],
        "_recal_italy_detail": recal_italy,
    }


def write_markdown(rows: list, meta: dict, path: Path) -> None:
    lines = [
        "# Step 1 -- Italy Threshold Recalibration",
        "",
        f"Generated: {meta['timestamp_utc']}",
        "",
        "Nested leave-one-fold-out threshold selection (grid search, F1-maximizing, "
        "0.01-0.99 step 0.01) applied to Italy's out-of-fold predictions from the pooled "
        "CV baseline. Every fold's reported metric uses a threshold selected from the "
        "OTHER 4 folds in its repeat only -- never from itself -- so these numbers are a "
        "legitimate estimate of a pre-committed threshold's performance on unseen Italy "
        "patients, not an optimistic in-sample best case.",
        "",
        "India is left at the fixed 0.5 threshold throughout: India's deficit is AUC "
        "(discrimination), not calibration, so recalibrating its threshold would not "
        "address the actual problem -- see `outputs/baseline/step1_baseline.md`.",
        "",
        "## Results, ranked by F1 improvement (descending)",
        "",
        "| Combo | Italy F1 @0.5 | Italy F1 recal. | ΔF1 | ΔPrecision | ΔRecall | "
        "Threshold (mean±SD) | Overall F1 @0.5 | Overall F1 mixed policy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["delta_italy_f1"], reverse=True):
        lines.append(
            f"| {r['combo'].replace('_v2_clean', '')} "
            f"| {r['baseline_italy_f1']:.3f} "
            f"| {r['recal_italy_f1']:.3f} "
            f"| {r['delta_italy_f1']:+.3f} "
            f"| {r['delta_italy_precision']:+.3f} "
            f"| {r['delta_italy_recall']:+.3f} "
            f"| {r['italy_threshold_mean']:.2f}±{r['italy_threshold_std']:.2f} "
            f"| {r['baseline_overall_f1']:.3f} "
            f"| {r['mixed_overall_f1']:.3f} |"
        )

    n = len(rows)
    n_improved = sum(1 for r in rows if r["delta_italy_f1"] > 0)
    n_naive_improved = sum(1 for r in rows if r["naive_leaky_italy_f1"] > r["baseline_italy_f1"])
    median_delta = float(np.median([r["delta_italy_f1"] for r in rows]))
    mean_delta = float(np.mean([r["delta_italy_f1"] for r in rows]))
    median_threshold = float(np.median([r["italy_threshold_mean"] for r in rows]))
    lines += [
        "",
        "## Summary",
        "",
        f"- {n_improved}/{n} combos improved Italy F1 under honest nested recalibration.",
        f"- Median delta-F1 (Italy, nested): {median_delta:+.3f}; mean: {mean_delta:+.3f}",
        f"- Median recalibrated threshold (Italy): {median_threshold:.2f} (baseline was fixed at 0.5)",
        "",
        "## Naive (leaky) vs. honest (nested) -- documented negative control",
        "",
        "Selecting a single F1-maximizing threshold from ALL of a combo's Italy out-of-fold "
        "predictions and then evaluating it on that SAME data (no nesting -- exactly the mistake "
        "this module's design exists to avoid) makes it LOOK like recalibration helps in "
        f"**{n_naive_improved}/{n}** combos. Once evaluated honestly with leave-one-fold-out nesting "
        f"above, only **{n_improved}/{n}** actually improve, and the median effect is "
        f"{'positive' if median_delta > 0 else 'slightly negative'} ({median_delta:+.3f}). This gap "
        "is measured directly on this project's own data, not asserted from general principle -- see "
        "`naive_leaky_italy_f1` in the CSV/JSON for the per-combo naive numbers. Conclusion: most of "
        "the apparent gain from a fixed post-hoc threshold shift on Italy is overfitting to which "
        "~104 Italy patients happened to be in the pool, not a real, generalizable improvement -- "
        "consistent with the small per-fold sample size (~20 Italy patients held out per fold).",
        "",
        "## Deployment threshold vs. reported metric -- do not conflate these",
        "",
        "The `italy_deployment_threshold` column (CSV/JSON only, not shown above) is selected "
        "once from ALL of a combo's Italy out-of-fold predictions pooled together, with no "
        "nesting. It is what you would actually configure a deployed model with -- but it is "
        "NOT a valid basis for the F1 numbers reported here, since it is chosen on the same "
        "data it would then be tested on. The nested procedure above intentionally produces up "
        "to 25 different per-fold thresholds rather than one, because that is what an honest "
        "leave-one-out estimate requires; the single deployment threshold is a separate, "
        "practical output with a different (weaker) evidentiary status.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Nested per-country (Italy) threshold recalibration.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--recal-dir", type=Path, default=RECAL_DIR)
    args = parser.parse_args()

    combos = _discover_real_combos(args.outputs_dir)
    if not combos:
        print(f"No completed combos found under {args.outputs_dir}.")
        return 1

    rows = [process_combo(args.outputs_dir, c) for c in combos]

    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "n_combos": len(rows)}
    args.recal_dir.mkdir(parents=True, exist_ok=True)

    csv_cols = [k for k in rows[0] if not k.startswith("_")]
    df_out = pd.DataFrame(rows)[csv_cols].sort_values("delta_italy_f1", ascending=False)
    csv_path = args.recal_dir / "italy_threshold_recalibration.csv"
    df_out.to_csv(csv_path, index=False)

    json_path = args.recal_dir / "italy_threshold_recalibration.json"
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "combos": rows}, f, indent=2)

    md_path = args.recal_dir / "italy_threshold_recalibration.md"
    write_markdown(rows, meta, md_path)

    print(f"\n{'=' * 100}\nItaly threshold recalibration -- {len(rows)} combos\n{'=' * 100}")
    print(
        df_out[
            ["combo", "baseline_italy_f1", "recal_italy_f1", "delta_italy_f1", "italy_threshold_mean", "italy_threshold_std"]
        ].to_string(index=False)
    )
    n_improved = int((df_out["delta_italy_f1"] > 0).sum())
    print(f"\n{n_improved}/{len(df_out)} combos improved. Median delta-F1 = {df_out['delta_italy_f1'].median():+.3f}")
    print(f"\nWrote {csv_path}\n      {json_path}\n      {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
