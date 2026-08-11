"""
Step 1 -- Measurement Harness: exact best-F1 threshold leaderboard.

Companion to threshold_recalibration.py, but answering a different, narrower
question: not "does recalibrating generalize to unseen patients" (the nested
leave-one-fold-out analysis in that module -- answer: not reliably, see
outputs/threshold_recalibration/), but simply "what IS the F1-maximizing
threshold for Italy, exactly, given every Italy prediction this harness has
produced." This is the deployment-style number: useful for configuring an
actual operating point, not for claiming a validated performance estimate.

EXACT vs. grid search: threshold_recalibration.py's grid search steps by
0.01, which can miss the true F1-maximizing point since F1 is a step
function that only changes value exactly at each observed predicted
probability. This module instead searches the midpoints between every pair
of consecutive sorted unique probabilities (plus the two boundary
candidates below the min and above the max) -- this is provably the exact
global optimum over the given data, not an approximation. Verified against
the coarser grid: the exact search never does worse and sometimes finds a
small additional gain (up to +0.006 F1 across the 18 combos), confirming
the grid was already close but not exact.

Design, matching what was already reported to and confirmed with the
project author: all of a country's out-of-fold predictions (every repeat,
every fold) are pooled into one set -- no nesting, no per-repeat averaging.
Italy uses its own exact best threshold; India stays at the fixed 0.5
threshold (India's deficit is AUC/discrimination, not calibration -- see
outputs/baseline/step1_baseline.md and threshold_recalibration.py's
docstring); Overall combines both countries' predictions under this mixed
policy. AUC (Italy/India/Overall) is reused as-is from step1_baseline.json
-- it is threshold-independent, so it is unaffected by any of this.

Usage:
    python classification/step1_cv_harness/best_threshold_leaderboard.py
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

from cv_config import BASELINE_DIR, OUTPUTS_DIR  # noqa: E402

BEST_THRESHOLD_DIR = OUTPUTS_DIR / "best_threshold"
INDIA_FIXED_THRESHOLD = 0.5


def exact_best_threshold_for_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    """The exact F1-maximizing threshold over (labels, probs): candidates are
    the midpoints between every pair of consecutive sorted unique
    probabilities, plus one candidate below the minimum (classifies
    everyone positive) and one above the maximum (classifies everyone
    negative). F1 cannot change value between two consecutive candidates,
    since no data point's classification flips in that interval -- so this
    set of candidates provably contains the global optimum, unlike a fixed-
    step grid. Ties broken by the median of every threshold tied for the
    best F1."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    uniq = np.unique(probs)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate([[uniq[0] - 1e-6], mids, [uniq[-1] + 1e-6]])

    best_f1 = -1.0
    tied = []
    for t in candidates:
        preds = (probs > t).astype(float)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            tied = [t]
        elif abs(f1 - best_f1) <= 1e-12:
            tied.append(t)
    return float(np.median(tied))


def _prf(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def process_combo(outputs_dir: Path, combo: str) -> dict:
    df = pd.read_csv(outputs_dir / combo / "oof_predictions.csv")
    italy = df[df["country"] == "Italy"]
    india = df[df["country"] == "India"]

    italy_labels, italy_probs = italy["label"].to_numpy(), italy["prob"].to_numpy()
    t_italy = exact_best_threshold_for_f1(italy_labels, italy_probs)
    italy_preds = (italy_probs > t_italy).astype(float)
    italy_metrics = _prf(italy_labels, italy_preds)

    india_labels, india_probs = india["label"].to_numpy(), india["prob"].to_numpy()
    india_preds = (india_probs > INDIA_FIXED_THRESHOLD).astype(float)
    india_metrics = _prf(india_labels, india_preds)

    overall_labels = np.concatenate([italy_labels, india_labels])
    overall_preds = np.concatenate([italy_preds, india_preds])
    overall_metrics = _prf(overall_labels, overall_preds)

    return {
        "combo": combo,
        "italy_threshold": t_italy,
        "italy_precision": italy_metrics["precision"],
        "italy_recall": italy_metrics["recall"],
        "italy_f1": italy_metrics["f1"],
        "india_threshold": INDIA_FIXED_THRESHOLD,
        "india_precision": india_metrics["precision"],
        "india_recall": india_metrics["recall"],
        "india_f1": india_metrics["f1"],
        "overall_precision": overall_metrics["precision"],
        "overall_recall": overall_metrics["recall"],
        "overall_f1": overall_metrics["f1"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact best-F1 Italy threshold leaderboard.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--dest-dir", type=Path, default=BEST_THRESHOLD_DIR)
    args = parser.parse_args()

    baseline = json.loads((args.baseline_dir / "step1_baseline.json").read_text())
    auc_by_combo = {c["combo"]: c for c in baseline["combos"]}

    rows = []
    for combo in sorted(auc_by_combo):
        r = process_combo(args.outputs_dir, combo)
        auc = auc_by_combo[combo]
        r["italy_auc"] = auc["italy_auc"]
        r["india_auc"] = auc["india_auc"]
        r["overall_auc"] = auc["overall_auc"]
        rows.append(r)

    args.dest_dir.mkdir(parents=True, exist_ok=True)
    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "n_combos": len(rows)}

    json_path = args.dest_dir / "italy_best_threshold_exact.json"
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "combos": rows}, f, indent=2)

    df_out = pd.DataFrame(rows).sort_values("italy_f1", ascending=False)
    csv_path = args.dest_dir / "italy_best_threshold_exact.csv"
    df_out.to_csv(csv_path, index=False)

    print(f"\n{'=' * 100}\nExact best-F1 Italy threshold -- {len(rows)} combos\n{'=' * 100}")
    print(df_out[["combo", "italy_threshold", "italy_precision", "italy_recall", "italy_f1"]].to_string(index=False))
    print(f"\nWrote {csv_path}\n      {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
