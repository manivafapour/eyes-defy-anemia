"""
Shared K-fold re-training engine for the 12 segmentation combos that
completed the Optuna sweep (Segmentation/scripts/train_pretrained/) --
see Segmentation/.project_memory/05_kfold_reevaluation.md for the full
rationale. Deliberately NOT another Optuna search: each combo already has
a best (learning_rate, weight_decay, loss_fn) from its study_summary.json,
and re-tuning here would confound "more robust measurement" with "better
hyperparameters" (same reasoning classification/step1_cv_harness/ already
documented for the identical situation).

Differences from trainer_engine.py, all deliberate (project author's
explicit request):
- 3-fold StratifiedKFold (stratified on country) over the pooled train+val
  patients, not a single fixed train/val split -- the held-out TEST split
  stays completely sealed, exactly mirroring
  classification/step1_cv_harness/cv_data.py's load_pool()/load_held_out().
- Fixed hyperparameters per combo (read by the generated entry-point
  script from that combo's real study_summary.json), not Optuna-sampled.
- MAX_EPOCHS=250 / EARLY_STOPPING_PATIENCE=7 (vs. trainer_engine.py's
  30/5) -- these exact values already exist elsewhere in this repo as
  classification's own live-trainer/CV-harness convention, not invented
  here.
- A ReduceLROnPlateau LR scheduler (trainer_engine.py has none at all),
  mirroring classification/datapreparepipeline/efficientnet_b0_forniceal_
  5fold_cv/cv_trainer_engine.py's exact hyperparameters.
- BATCH_SIZE=32 (vs. trainer_engine.py's 16), independent constant --
  does not touch dataset.py's BATCH_SIZE, which the Optuna sweep scripts
  still use unchanged.
- Only 1 checkpoint per fold (fp16, via trainer_engine._half_state_dict),
  not per-loss-function variants -- there's only one fixed loss function
  per combo now, so that distinction doesn't apply here.

Everything else (train/eval loop internals, metrics, the final TEST-set
evaluation, fp16 checkpoint saving, param/latency measurement) is
imported and reused from trainer_engine.py and segmentation_metrics.py
unchanged -- not duplicated.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    ALIGNED_TISSUE_CONFIG,
    SPLITS_CSV,
    AlignedConjunctivaSegmentationDataset,
    get_eval_transforms,
    get_train_transforms,
)
from segmentation_metrics import evaluate_final_test_set  # noqa: E402
from segmentation_plots import generate_kfold_plots  # noqa: E402
from trainer_engine import (  # noqa: E402
    CHECKPOINTS_DIR,
    DEVICE,
    LOGS_DIR,
    LOSS_REGISTRY,
    PLOTS_DIR,
    _half_state_dict,
    count_parameters,
    evaluate,
    measure_inference_latency,
    train_one_epoch,
)

# --------------------------------------------------------------------------
# Configuration -- independent from trainer_engine.py's constants (that
# module's MAX_EPOCHS=30/EARLY_STOPPING_PATIENCE=5/BATCH_SIZE=16 keep
# governing the Optuna sweep scripts unchanged); see module docstring.
# --------------------------------------------------------------------------
N_FOLDS = 3
MAX_EPOCHS = 250
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
MIN_LR = 1e-6
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42


def _load_pool_df(tissue_type: str) -> pd.DataFrame:
    """train+val patients only (the K-fold pool) -- test stays sealed,
    mirroring classification/step1_cv_harness/cv_data.py's load_pool()/
    load_held_out() split exactly. Filtered to patients with a successful
    alignment for this tissue_type (the same join
    AlignedConjunctivaSegmentationDataset does internally), so every fold
    only ever contains patients the dataset class can actually load."""
    tissue_config = ALIGNED_TISSUE_CONFIG[tissue_type]
    splits_df = pd.read_csv(SPLITS_CSV)
    pool_df = splits_df[splits_df["split"].isin(["train", "val"])]

    alignment_log = pd.read_csv(tissue_config["alignment_log_csv"])
    aligned_ids = set(alignment_log.loc[alignment_log["status"] == "ok", "patient_id"])
    pool_df = pool_df[pool_df["patient_id"].isin(aligned_ids)]

    return pool_df.reset_index(drop=True)


def run_kfold_study(
    model_name: str,
    build_model,
    tissue_type: str,
    image_size: int,
    learning_rate: float,
    weight_decay: float,
    loss_fn_name: str,
    n_folds: int = N_FOLDS,
) -> dict:
    """Trains `build_model()` under n_folds-fold StratifiedKFold (stratified
    on country) over the pooled train+val patients for `tissue_type`, using
    the given FIXED hyperparameters for every fold (no per-fold tuning).
    Each fold's best-val-Dice checkpoint (fp16) is evaluated once against
    the sealed TEST split via the unchanged evaluate_final_test_set().
    Persists per-fold and aggregated (mean/std across folds) results plus
    plots, mirroring trainer_engine.run_study()'s output contract but for
    K folds instead of one run. Returns the list of per-fold result dicts."""
    print(f"Using device: {DEVICE}")
    print(f"Model: {model_name} (K-fold re-training, fixed hyperparameters)")
    print(f"tissue_type={tissue_type} image_size={image_size}")
    print(f"learning_rate={learning_rate} weight_decay={weight_decay} loss_fn={loss_fn_name}")
    print(
        f"n_folds={n_folds} MAX_EPOCHS={MAX_EPOCHS} "
        f"EARLY_STOPPING_PATIENCE={EARLY_STOPPING_PATIENCE} BATCH_SIZE={BATCH_SIZE}"
    )

    pool_df = _load_pool_df(tissue_type)
    patient_ids = pool_df["patient_id"].to_numpy()
    countries = pool_df["country"].to_numpy()
    print(f"Pool (train+val, aligned): {len(patient_ids)} patients")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, countries)):
        fold_num = fold_idx + 1
        train_ids = patient_ids[train_idx].tolist()
        val_ids = patient_ids[val_idx].tolist()
        print(f"\n=== {model_name} | Fold {fold_num}/{n_folds}: train={len(train_ids)} val={len(val_ids)} ===")

        train_dataset = AlignedConjunctivaSegmentationDataset(
            split="train",
            splits_csv=SPLITS_CSV,
            tissue_type=tissue_type,
            transform=get_train_transforms(image_size),
            patient_ids=train_ids,
        )
        val_dataset = AlignedConjunctivaSegmentationDataset(
            split="val",
            splits_csv=SPLITS_CSV,
            tissue_type=tissue_type,
            transform=get_eval_transforms(image_size),
            patient_ids=val_ids,
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = build_model().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR
        )
        criterion = LOSS_REGISTRY[loss_fn_name]()

        checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}_fold{fold_num}.pth"
        best_val_loss = float("inf")
        best_val_dice = 0.0
        epochs_without_improvement = 0
        epoch_history = []

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "lr": current_lr,
                }
            )

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                torch.save(_half_state_dict(model.state_dict()), checkpoint_path)

            print(
                f"[{model_name} | fold {fold_num}/{n_folds}] Epoch {epoch:>3}/{MAX_EPOCHS} - "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f} "
                f"val_iou={val_metrics['iou']:.4f} lr={current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"[{model_name} | fold {fold_num}] Early stopping at epoch {epoch} "
                        f"(no val_loss improvement for {EARLY_STOPPING_PATIENCE} epochs)."
                    )
                    break

        test_df = None
        if checkpoint_path.exists():
            print(f"--- Fold {fold_num} test-set evaluation ({checkpoint_path.name}) ---")
            test_model = build_model().to(DEVICE)
            test_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            test_df = evaluate_final_test_set(
                test_model,
                AlignedConjunctivaSegmentationDataset,
                SPLITS_CSV,
                DEVICE,
                get_eval_transforms(image_size),
                tissue_type=tissue_type,
            )
            test_df["fold"] = fold_num
            print(
                f"Fold {fold_num} test Dice={test_df['dice'].mean():.4f} "
                f"IoU={test_df['iou'].mean():.4f} HD95={test_df['hd95'].mean():.2f}px"
            )
            del test_model
        else:
            print(f"No checkpoint saved for fold {fold_num} -- skipping its test-set evaluation.")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        fold_results.append(
            {
                "fold": fold_num,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "best_val_dice": best_val_dice,
                "n_epochs_run": len(epoch_history),
                "epoch_history": epoch_history,
                "test_df": test_df,
                "checkpoint_path": str(checkpoint_path),
            }
        )

    print("\n--- Measuring architecture properties (params, inference latency) ---")
    probe_model = build_model().to(DEVICE)
    n_params = count_parameters(probe_model)
    latency_ms = measure_inference_latency(probe_model, image_size, DEVICE)
    print(f"Parameters: {n_params:,}")
    print(f"Inference latency (batch_size=1, {image_size}x{image_size}): {latency_ms:.2f} ms")
    del probe_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    _save_kfold_outputs(
        model_name, fold_results, n_params, latency_ms, image_size,
        tissue_type, learning_rate, weight_decay, loss_fn_name, n_folds,
    )
    return fold_results


def _save_kfold_outputs(
    model_name, fold_results, n_params, latency_ms, image_size,
    tissue_type, learning_rate, weight_decay, loss_fn_name, n_folds,
):
    """Persists per-fold and cross-fold-aggregated results, mirroring
    trainer_engine._save_outputs()'s output contract (JSON summary + CSVs
    + plots) but for K folds instead of one Optuna study."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    test_dfs = [fr["test_df"] for fr in fold_results if fr["test_df"] is not None]
    combined_test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else None

    folds_summary_rows = []
    for fr in fold_results:
        row = {
            "fold": fr["fold"],
            "n_train": fr["n_train"],
            "n_val": fr["n_val"],
            "best_val_dice": fr["best_val_dice"],
            "n_epochs_run": fr["n_epochs_run"],
            "checkpoint_path": fr["checkpoint_path"],
        }
        if fr["test_df"] is not None:
            row["test_dice"] = fr["test_df"]["dice"].mean()
            row["test_iou"] = fr["test_df"]["iou"].mean()
            row["test_precision"] = fr["test_df"]["precision"].mean()
            row["test_recall"] = fr["test_df"]["recall"].mean()
            row["test_hd95"] = fr["test_df"]["hd95"].mean()
        folds_summary_rows.append(row)
    folds_df = pd.DataFrame(folds_summary_rows)

    folds_csv_path = LOGS_DIR / f"{model_name}_kfold_folds.csv"
    folds_df.to_csv(folds_csv_path, index=False)
    print(f"\nSaved per-fold summary to {folds_csv_path}")

    summary = {
        "model_name": model_name,
        "tissue_type": tissue_type,
        "n_folds": n_folds,
        "n_params": n_params,
        "inference_latency_ms_batch1": latency_ms,
        "image_size": image_size,
        "fixed_hyperparameters": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss_fn": loss_fn_name,
        },
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "factor": SCHEDULER_FACTOR,
            "patience": SCHEDULER_PATIENCE,
            "min_lr": MIN_LR,
        },
        "per_fold": folds_summary_rows,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if "test_dice" in folds_df.columns:
        metric_cols = ["test_dice", "test_iou", "test_precision", "test_recall", "test_hd95"]
        summary["aggregate_across_folds"] = {
            "mean": {c.replace("test_", ""): float(folds_df[c].mean()) for c in metric_cols},
            "std": {c.replace("test_", ""): float(folds_df[c].std()) for c in metric_cols},
        }
        print("\n--- Aggregate across folds (mean +/- std) ---")
        for c in metric_cols:
            print(f"  {c.replace('test_', ''):10s} = {folds_df[c].mean():.4f} +/- {folds_df[c].std():.4f}")

    test_metrics_summary = None
    if combined_test_df is not None and len(combined_test_df):
        metric_cols = ["dice", "iou", "precision", "recall"]
        overall = combined_test_df[metric_cols].mean(numeric_only=True).to_dict()
        overall["hd95"] = float(np.nanmean(combined_test_df["hd95"]))
        n_hd95_undefined = int(combined_test_df["hd95"].isna().sum())
        per_country = combined_test_df.groupby("country")[metric_cols].mean(numeric_only=True)
        per_country["hd95"] = combined_test_df.groupby("country")["hd95"].apply(lambda s: float(np.nanmean(s)))

        test_metrics_summary = {
            "n_test_patients": combined_test_df["patient_id"].nunique(),
            "n_test_evaluations": len(combined_test_df),
            "n_hd95_undefined_empty_mask": n_hd95_undefined,
            "overall": {k: float(v) for k, v in overall.items()},
            "by_country": per_country.to_dict(orient="index"),
        }
        summary["test_set_metrics_pooled_across_folds"] = test_metrics_summary

        test_csv_path = LOGS_DIR / f"{model_name}_kfold_test_per_patient.csv"
        combined_test_df.to_csv(test_csv_path, index=False)
        print(f"Saved per-patient-per-fold test metrics to {test_csv_path}")

    summary_json_path = LOGS_DIR / f"{model_name}_kfold_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved K-fold summary to {summary_json_path}")

    fold_histories = [fr["epoch_history"] for fr in fold_results]
    written_plots = generate_kfold_plots(
        PLOTS_DIR, model_name, fold_histories, folds_df, test_metrics_summary, combined_test_df
    )
    if written_plots:
        print(f"Saved {len(written_plots)} plots to {PLOTS_DIR}:")
        for p in written_plots:
            print(f"  {p.name}")
