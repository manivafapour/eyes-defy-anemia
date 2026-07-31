"""
EfficientNet-B0 (forniceal_palpebral) -- 5-Fold Cross-Validation pipeline.
Main entry point.

Orchestrates: load the 178-patient CV pool (33-patient test set locked
away, never touched) -> build 5 stratified folds -> train each fold from a
fresh random head init with fixed, Batch-1-derived hyperparameters,
gradient clipping, ReduceLROnPlateau, and EarlyStopping -> aggregate
results across folds into a single cv_summary.json.

Three ways to run this script:
  python run_cv_training.py               -- all 5 folds + aggregation, one process (local use)
  python run_cv_training.py --fold N       -- only fold N (1-5), no aggregation
  python run_cv_training.py --aggregate    -- no training, aggregate already-saved fold results

The --fold / --aggregate split exists specifically for the Kaggle notebook:
running each fold as its own `!python` cell (with a zip-and-save step after
each) means a mid-run session interruption only loses the fold in
progress, not the whole 5-fold run -- same incremental-safety principle
used for classification-final-fixed.ipynb (batch 1) and
classification-batch2.ipynb. The default (no-args) full-run path is kept
for local/non-Kaggle use and is unchanged from the original, already
dry-run-verified behavior.
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

from cv_dataset import load_cv_pool, build_folds, N_FOLDS, SEED
from cv_trainer_engine import (
    run_fold,
    MODEL_NAME,
    LOGS_DIR,
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DROPOUT_RATE,
    BATCH_SIZE,
    MAX_EPOCHS,
    GRAD_CLIP_MAX_NORM,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MIN_LR,
    EARLY_STOPPING_PATIENCE,
)


def _mean_std(values: list) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "std": None}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def aggregate_fold_results(fold_results: list) -> dict:
    """Mean +/- std across the 5 folds, overall and per country -- the
    actual point of doing CV instead of a single split: a variance-aware
    estimate rather than one (possibly lucky or unlucky) split's number."""
    buckets = ["overall", "India", "Italy"]
    metric_keys = ["accuracy", "precision", "recall", "sensitivity", "specificity", "balanced_accuracy", "f1", "auc"]

    aggregate = {}
    for bucket in buckets:
        aggregate[bucket] = {}
        for key in metric_keys:
            values = [r["best_val_metrics"][bucket].get(key) for r in fold_results]
            aggregate[bucket][key] = _mean_std(values)

    aggregate["n_epochs_run"] = _mean_std([r["n_epochs_run"] for r in fold_results])
    return aggregate


def _print_run_header() -> None:
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(
        f"Locked hyperparameters: lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, "
        f"dropout={DROPOUT_RATE} (from Batch 1 trial #9, no Optuna)"
    )
    print(
        f"batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, grad_clip_max_norm={GRAD_CLIP_MAX_NORM}, "
        f"scheduler=ReduceLROnPlateau(factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE}, min_lr={MIN_LR}), "
        f"early_stopping_patience={EARLY_STOPPING_PATIENCE}"
    )


def _save_summary(fold_results: list, pool_size: int) -> None:
    """Shared by both the full-run path and --aggregate, so the two can
    never silently diverge in what they compute or how they save it."""
    aggregate = aggregate_fold_results(fold_results)

    print(f"\n{'=' * 70}\n5-Fold CV Summary -- {MODEL_NAME}\n{'=' * 70}")
    for bucket in ["overall", "India", "Italy"]:
        f1 = aggregate[bucket]["f1"]
        bacc = aggregate[bucket]["balanced_accuracy"]
        auc = aggregate[bucket]["auc"]
        print(
            f"  {bucket:8s}: F1={f1['mean']:.4f}+/-{f1['std']:.4f}  "
            f"BalAcc={bacc['mean']:.4f}+/-{bacc['std']:.4f}  "
            f"AUC={auc['mean'] if auc['mean'] is not None else float('nan'):.4f}"
            f"+/-{auc['std'] if auc['std'] is not None else float('nan'):.4f}"
        )

    summary = {
        "model_name": MODEL_NAME,
        "n_folds": N_FOLDS,
        "cv_pool_size": pool_size,
        "test_set_excluded": True,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
            "source": "Batch 1 trial #9 (efficientnet_b0_forniceal_palpebral_v2), Optuna not used here",
        },
        "training_config": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "min_lr": MIN_LR,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "per_fold_results": [
            {
                "fold": r["fold"],
                "n_epochs_run": r["n_epochs_run"],
                "best_val_f1": r["best_val_f1"],
                "best_val_metrics": r["best_val_metrics"],
                "checkpoint_path": r["checkpoint_path"],
            }
            for r in fold_results
        ],
        "aggregate": aggregate,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOGS_DIR / f"{MODEL_NAME}_cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved cross-fold summary to {summary_path}")


def train_single_fold(fold_num: int) -> None:
    """Runs exactly one fold (1-5) and saves its checkpoint/history/plots --
    nothing else. Used by the Kaggle notebook's per-fold cells."""
    if not 1 <= fold_num <= N_FOLDS:
        raise ValueError(f"fold_num must be between 1 and {N_FOLDS}, got {fold_num}")

    _print_run_header()
    pool = load_cv_pool()
    print(f"\nCV pool: {len(pool)} patients (33-patient test set excluded, never loaded)")
    folds = build_folds(pool, n_folds=N_FOLDS, seed=SEED)

    train_df, val_df = folds[fold_num - 1]
    run_fold(fold_num, train_df, val_df)


def aggregate_only() -> None:
    """Loads the per-fold history JSONs already saved on disk (one per
    fold, from prior train_single_fold calls) and computes the cross-fold
    summary -- no training. Fails loudly, not silently, if any fold's
    history is missing rather than aggregating a partial/misleading set."""
    fold_results = []
    missing = []
    for i in range(1, N_FOLDS + 1):
        history_path = LOGS_DIR / f"{MODEL_NAME}_fold{i}_history.json"
        if not history_path.exists():
            missing.append(str(history_path))
            continue
        with open(history_path) as f:
            fold_results.append(json.load(f))

    if missing:
        raise FileNotFoundError(
            f"Cannot aggregate -- {len(missing)} fold history file(s) missing: {missing}. "
            f"Run all {N_FOLDS} folds first (--fold N for each, 1 through {N_FOLDS})."
        )

    pool = load_cv_pool()
    _save_summary(fold_results, pool_size=len(pool))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=None, help="Run only this fold (1-5), skip aggregation.")
    parser.add_argument(
        "--aggregate", action="store_true", help="Skip training; aggregate already-saved per-fold results only."
    )
    args = parser.parse_args()

    if args.aggregate:
        aggregate_only()
        return

    if args.fold is not None:
        train_single_fold(args.fold)
        return

    # No args: original full-run behavior -- all 5 folds + aggregation in one process.
    _print_run_header()
    pool = load_cv_pool()
    print(f"\nCV pool: {len(pool)} patients (33-patient test set excluded, never loaded)")

    folds = build_folds(pool, n_folds=N_FOLDS, seed=SEED)
    print(f"Built {len(folds)} stratified folds (country x anemic_label)")
    for i, (train_df, val_df) in enumerate(folds, start=1):
        print(f"  Fold {i}: train={len(train_df)} val={len(val_df)}")

    fold_results = []
    for i, (train_df, val_df) in enumerate(folds, start=1):
        result = run_fold(i, train_df, val_df)
        fold_results.append(result)

    _save_summary(fold_results, pool_size=len(pool))


if __name__ == "__main__":
    main()
