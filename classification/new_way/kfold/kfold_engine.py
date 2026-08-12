"""
Fixed-hyperparameter 3-fold re-training engine for the 6 best combos from
the new_way/ 16-combo Optuna sweep (Output/version1/). See
classification/.project_memory/12_new_way_kfold_retraining.md for the full
rationale; the short version:

Each of the 16 combos' reported number is a single point estimate -- one
12-trial Optuna search picked one winning hyperparameter set and trained
once. This engine re-trains the 6 best of those combos, using each one's
OWN already-found best (learning_rate, weight_decay, dropout_rate) -- no
Optuna anywhere in this path, no re-tuning -- under 3-fold cross-validation
over the pooled train+val patients, so the final number is a mean +/- std
across 3 independent trainings rather than one lucky/unlucky draw.

Deliberately NOT built on trainer_engine.py's make_objective()/run_study()
-- those are Optuna-shaped (they sample hyperparameters via trial.suggest_*)
and this path has no trial at all. Reused from trainer_engine.py instead,
unchanged: ARCHITECTURE_REGISTRY (the build_fn/input_size registry),
compute_metrics(), and evaluate() (a pure function with no path
dependency, safe to import as-is). train_one_epoch() is rewritten here to
add gradient clipping, which the shared version doesn't have and which
this pipeline deliberately does not add to the shared version either (that
would silently change behavior for the 16 already-run Optuna combos) --
exact same reasoning and the exact same fixed hyperparameter numbers as
classification/datapreparepipeline/efficientnet_b0_forniceal_5fold_cv/
cv_trainer_engine.py, this project's own most directly analogous
precedent (not segmentation's kfold_engine.py, which uses slightly
different numbers tuned for a different domain).

Data: TRAIN portion of each fold reads from new_way/balanced_dataset.py's
BalancedTissueClassificationDataset (offline country-stratified label
balance, WITH online augmentation still layered on top via
get_train_transforms -- both, per the project author's explicit
instruction, not one instead of the other). VAL and the final sealed TEST
read from datapreparepipeline/dataset.py's TissueClassificationDataset,
unmodified, deterministic eval transform only -- never augmented, offline
or online. Both dataset classes' patient_ids param (added alongside this
engine) is what lets a fold's train/val patient lists select from either.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

KFOLD_DIR = Path(__file__).resolve().parent  # classification/new_way/kfold/
NEW_WAY_DIR = KFOLD_DIR.parent  # classification/new_way/
CLASSIFICATION_DIR = NEW_WAY_DIR.parent  # classification/
DATAPREPAREPIPELINE_DIR = CLASSIFICATION_DIR / "datapreparepipeline"

sys.path.insert(0, str(DATAPREPAREPIPELINE_DIR))
sys.path.insert(0, str(NEW_WAY_DIR))

from trainer_engine import ARCHITECTURE_REGISTRY, DEVICE, compute_metrics, evaluate  # noqa: E402 -- reuse, don't duplicate
from dataset import (  # noqa: E402
    EXTRACTION_LOG_CSV,
    SPLITS_CSV,
    TissueClassificationDataset,
    get_eval_transforms,
    get_train_transforms,
)
from balanced_dataset import BalancedTissueClassificationDataset  # noqa: E402

# --------------------------------------------------------------------------
# Configuration -- independent of trainer_engine.py's Optuna-sweep constants
# (that module's MAX_EPOCHS=250/EARLY_STOPPING_PATIENCE=7/BATCH_SIZE=32
# already happen to match some of these, but every constant below is its
# own independent value, not inherited).
# --------------------------------------------------------------------------
N_FOLDS = 3
MAX_EPOCHS = 250
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

# Mirrors efficientnet_b0_forniceal_5fold_cv/cv_trainer_engine.py's own
# fixed-hyperparameter CV numbers exactly -- classification's own most
# directly analogous precedent for this exact scenario.
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
MIN_LR = 1e-6
EARLY_STOPPING_PATIENCE = 15  # > 2x SCHEDULER_PATIENCE, so the scheduler gets a real chance to act first
GRAD_CLIP_MAX_NORM = 1.0

TISSUE_TYPES = ["palpebral", "forniceal_palpebral"]
COUNTRIES = ["India", "Italy"]

OUTPUTS_DIR = NEW_WAY_DIR / "Output" / "version2"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"


def _half_state_dict(state_dict: dict) -> dict:
    """fp16 checkpoint saving -- halves file size (6 architectures x 3
    folds is ~6.9GB at fp32, ~3.5GB at fp16). Loading a fp16 state_dict
    into a fresh fp32 model works with no special-casing (Tensor.copy_
    inside load_state_dict casts automatically) -- already proven by
    Segmentation/scripts/train_pretrained_kfold/kfold_engine.py's
    identical _half_state_dict helper; not imported from there (that
    would cross this project's phase-isolation boundary), just the same
    one-line idea reimplemented locally."""
    return {k: v.half() for k, v in state_dict.items()}


# --------------------------------------------------------------------------
# Pool + fold construction
# --------------------------------------------------------------------------
def load_pool_df(tissue_type: str) -> pd.DataFrame:
    """train+val patients only (the K-fold pool) -- test stays sealed,
    mirroring every CV precedent in this repo (step1_cv_harness/cv_data.py,
    efficientnet_b0_forniceal_5fold_cv/cv_dataset.py, segmentation's
    kfold_engine.py)."""
    if tissue_type not in TISSUE_TYPES:
        raise ValueError(f"tissue_type must be one of {TISSUE_TYPES}, got {tissue_type!r}")
    splits = pd.read_csv(SPLITS_CSV)
    log = pd.read_csv(EXTRACTION_LOG_CSV)
    ok_ids = set(log.loc[log[f"{tissue_type}_status"] == "ok", "patient_id"])
    pool = splits[splits["split"].isin(["train", "val"])]
    pool = pool[pool["patient_id"].isin(ok_ids)]
    return pool.reset_index(drop=True)


def build_folds(pool_df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> list:
    """Patient-level StratifiedKFold on the country+label compound key --
    same stratification discipline as this project's original train/val/
    test split and every classification CV precedent (efficientnet_b0_
    forniceal_5fold_cv/cv_dataset.py's build_folds(), step1_cv_harness's
    stratum key). NOT segmentation's country-only key -- here the label
    itself is the prediction target, so it belongs in the stratum.
    Returns a list of n_folds (train_ids, val_ids) tuples, deterministic
    given the same pool_df/seed -- call once per tissue_type and reuse the
    SAME result across every model that trains on that tissue_type, for a
    fair apples-to-apples comparison."""
    strata = pool_df["country"] + "_" + pool_df["anemic_label"].astype(int).astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(pool_df, strata):
        train_ids = pool_df.iloc[train_idx]["patient_id"].tolist()
        val_ids = pool_df.iloc[val_idx]["patient_id"].tolist()
        folds.append((train_ids, val_ids))
    return folds


# --------------------------------------------------------------------------
# Train loop -- evaluate() reused as-is from trainer_engine.py (pure
# function, no path dependency); train_one_epoch() rewritten here
# specifically to add gradient clipping, which the shared version doesn't
# have (see module docstring for why it isn't added there instead).
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_max_norm: float) -> float:
    model.train()
    running_loss = 0.0
    for images, labels, _countries in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


# --------------------------------------------------------------------------
# One fold
# --------------------------------------------------------------------------
def run_fold(
    fold_num: int,
    n_folds: int,
    model_name: str,
    arch_name: str,
    tissue_type: str,
    train_ids: list,
    val_ids: list,
    learning_rate: float,
    weight_decay: float,
    dropout_rate: float,
) -> dict:
    """Trains one fold with a FRESH, randomly-initialized head (every
    ARCHITECTURE_REGISTRY build_fn constructs a new head from scratch --
    no checkpoint is ever loaded here), fixed hyperparameters throughout,
    no Optuna. Train reads the offline-balanced+online-augmented data
    (patient_ids=train_ids); val reads real, unmodified, unaugmented
    images (patient_ids=val_ids)."""
    print(f"\n{'=' * 70}\n{model_name} | Fold {fold_num}/{n_folds} -- train={len(train_ids)} val={len(val_ids)}\n{'=' * 70}")

    torch.manual_seed(SEED + fold_num)  # reproducible but distinct per fold, not identically seeded across folds

    arch_config = ARCHITECTURE_REGISTRY[arch_name]
    image_size = arch_config["input_size"]

    train_dataset = BalancedTissueClassificationDataset(
        tissue_type=tissue_type, transform=get_train_transforms(image_size), patient_ids=train_ids
    )
    val_dataset = TissueClassificationDataset(
        split="val", tissue_type=tissue_type, transform=get_eval_transforms(image_size), patient_ids=val_ids
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    train_labels = train_dataset.df["anemic_label"].to_numpy()
    n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    print(f"[{model_name} | fold {fold_num}] train n={len(train_dataset)} (pos={n_pos:.0f} neg={n_neg:.0f} pos_weight={pos_weight.item():.3f}) val n={len(val_dataset)}")

    model = arch_config["build_fn"](dropout_rate).to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR
    )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / f"{model_name}_fold{fold_num}_best.pth"

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_val_metrics = None
    epochs_without_improvement = 0
    train_loss_history, val_loss_history, lr_history = [], [], []
    val_overall_metrics_history = []

    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, GRAD_CLIP_MAX_NORM)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        val_f1 = val_metrics["overall"]["f1"]
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        lr_history.append(current_lr)
        val_overall_metrics_history.append(
            {
                "accuracy": val_metrics["overall"]["accuracy"],
                "f1": val_metrics["overall"]["f1"],
                "sensitivity": val_metrics["overall"]["sensitivity"],
                "specificity": val_metrics["overall"]["specificity"],
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_metrics = val_metrics
            torch.save(_half_state_dict(model.state_dict()), checkpoint_path)
            print(f"[{model_name} | fold {fold_num}] New best val_f1={val_f1:.4f} -> saved {checkpoint_path.name}")

        print(
            f"[{model_name} | fold {fold_num}] Epoch {epoch:>3}/{MAX_EPOCHS} - "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} lr={current_lr:.2e}"
        )

        scheduler.step(val_loss)  # ReduceLROnPlateau -- must come after computing val_loss this epoch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"[{model_name} | fold {fold_num}] Early stopping at epoch {epoch}.")
                break

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # --- Final sealed-TEST evaluation, from a FRESH model loaded off the
    # saved best-val-F1 checkpoint (not whatever's left in memory after the
    # loop, which may be past its best point if early stopping fired). ---
    test_metrics = None
    if checkpoint_path.exists():
        print(f"[{model_name} | fold {fold_num}] --- Sealed test-set evaluation ({checkpoint_path.name}) ---")
        test_model = arch_config["build_fn"](dropout_rate).to(DEVICE)
        test_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        test_dataset = TissueClassificationDataset(
            split="test", tissue_type=tissue_type, transform=get_eval_transforms(image_size)
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        _test_loss, test_metrics = evaluate(test_model, test_loader, criterion, DEVICE)
        print(
            f"[{model_name} | fold {fold_num}] test F1={test_metrics['overall']['f1']:.4f} "
            f"acc={test_metrics['overall']['accuracy']:.4f} auc={test_metrics['overall'].get('auc')}"
        )
        del test_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"[{model_name} | fold {fold_num}] No checkpoint saved -- skipping test-set evaluation.")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    history = {
        "model_name": model_name,
        "fold": fold_num,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_epochs_run": epoch,
        "best_val_f1": best_val_f1,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "lr_history": lr_history,
        "checkpoint_path": str(checkpoint_path),
    }
    history_path = LOGS_DIR / f"{model_name}_fold{fold_num}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    _plot_loss_curve(model_name, fold_num, train_loss_history, val_loss_history)
    _plot_lr_curve(model_name, fold_num, lr_history)
    _plot_val_metrics_curve(model_name, fold_num, val_overall_metrics_history)
    if best_val_metrics is not None:
        _plot_confusion_matrices(model_name, fold_num, best_val_metrics)
        _plot_roc_curves(model_name, fold_num, best_val_metrics)

    print(f"[{model_name} | fold {fold_num}] Done. Best val_f1={best_val_f1:.4f} after {epoch} epochs. Saved {history_path.name}")
    return history


# --------------------------------------------------------------------------
# Full study: N_FOLDS folds for one (model_name, arch_name, tissue_type,
# fixed hyperparameters) combo, then cross-fold aggregation.
# --------------------------------------------------------------------------
def run_kfold_study(
    model_name: str,
    arch_name: str,
    tissue_type: str,
    learning_rate: float,
    weight_decay: float,
    dropout_rate: float,
    n_folds: int = N_FOLDS,
) -> list:
    if arch_name not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"arch_name must be one of {list(ARCHITECTURE_REGISTRY)}, got {arch_name!r}")

    print(f"Using device: {DEVICE}")
    print(f"Model: {model_name} (fixed-hyperparameter {n_folds}-fold CV, no Optuna)")
    print(f"arch_name={arch_name} tissue_type={tissue_type}")
    print(f"learning_rate={learning_rate} weight_decay={weight_decay} dropout_rate={dropout_rate}")
    print(f"MAX_EPOCHS={MAX_EPOCHS} EARLY_STOPPING_PATIENCE={EARLY_STOPPING_PATIENCE} BATCH_SIZE={BATCH_SIZE}")

    pool_df = load_pool_df(tissue_type)
    folds = build_folds(pool_df, n_folds=n_folds)
    print(f"Pool (train+val): {len(pool_df)} patients -> {n_folds} folds")

    fold_histories = []
    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        fold_num = fold_idx + 1
        history = run_fold(
            fold_num, n_folds, model_name, arch_name, tissue_type, train_ids, val_ids,
            learning_rate, weight_decay, dropout_rate,
        )
        fold_histories.append(history)

    _save_kfold_summary(
        model_name, fold_histories, n_folds, arch_name, tissue_type, learning_rate, weight_decay, dropout_rate
    )
    return fold_histories


def _save_kfold_summary(
    model_name: str,
    fold_histories: list,
    n_folds: int,
    arch_name: str,
    tissue_type: str,
    learning_rate: float,
    weight_decay: float,
    dropout_rate: float,
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    folds_rows = []
    for h in fold_histories:
        row = {
            "fold": h["fold"],
            "n_train": h["n_train"],
            "n_val": h["n_val"],
            "n_epochs_run": h["n_epochs_run"],
            "best_val_f1": h["best_val_f1"],
            "checkpoint_path": h["checkpoint_path"],
        }
        if h["test_metrics"] is not None:
            for bucket in ["overall"] + COUNTRIES:
                m = h["test_metrics"].get(bucket, {})
                prefix = "" if bucket == "overall" else f"{bucket.lower()}_"
                row[f"test_{prefix}f1"] = m.get("f1")
                row[f"test_{prefix}accuracy"] = m.get("accuracy")
                row[f"test_{prefix}auc"] = m.get("auc")
        folds_rows.append(row)
    folds_df = pd.DataFrame(folds_rows)
    folds_csv_path = LOGS_DIR / f"{model_name}_kfold_folds.csv"
    folds_df.to_csv(folds_csv_path, index=False)
    print(f"\nSaved per-fold summary to {folds_csv_path}")

    summary = {
        "model_name": model_name,
        "arch_name": arch_name,
        "tissue_type": tissue_type,
        "n_folds": n_folds,
        "fixed_hyperparameters": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
        },
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "scheduler": {"type": "ReduceLROnPlateau", "factor": SCHEDULER_FACTOR, "patience": SCHEDULER_PATIENCE, "min_lr": MIN_LR},
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "per_fold": folds_rows,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    metric_cols = [c for c in folds_df.columns if c.startswith("test_")]
    if metric_cols:
        summary["aggregate_across_folds"] = {
            "mean": {c: float(folds_df[c].mean()) for c in metric_cols if folds_df[c].notna().any()},
            "std": {c: float(folds_df[c].std()) for c in metric_cols if folds_df[c].notna().any()},
        }
        print("\n--- Aggregate test metrics across folds (mean +/- std) ---")
        for c in metric_cols:
            if folds_df[c].notna().any():
                print(f"  {c:24s} = {folds_df[c].mean():.4f} +/- {folds_df[c].std():.4f}")

    summary_json_path = LOGS_DIR / f"{model_name}_kfold_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved K-fold summary to {summary_json_path}")

    if metric_cols:
        _plot_kfold_test_summary(model_name, folds_df)


# --------------------------------------------------------------------------
# Plots -- adapted from cv_trainer_engine.py's own plot functions, but
# writing to THIS pipeline's own PLOTS_DIR (new_way/Output/version2/plots/,
# never classification/outputs/plots/ or new_way/Output/version1/plots/).
# --------------------------------------------------------------------------
def _plot_loss_curve(model_name, fold_num, train_loss_history, val_loss_history) -> Path:
    epochs = range(1, len(train_loss_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss_history, label="train loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss_history, label="val loss", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title(f"{model_name} -- fold {fold_num} -- train vs val loss")
    ax.legend()
    ax.grid(alpha=0.3)
    path = PLOTS_DIR / f"{model_name}_fold{fold_num}_loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_lr_curve(model_name, fold_num, lr_history) -> Path:
    epochs = range(1, len(lr_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, lr_history, marker="o", markersize=3, color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title(f"{model_name} -- fold {fold_num} -- ReduceLROnPlateau schedule")
    ax.grid(alpha=0.3)
    path = PLOTS_DIR / f"{model_name}_fold{fold_num}_lr_schedule.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_val_metrics_curve(model_name, fold_num, val_overall_metrics_history) -> Path:
    epochs = range(1, len(val_overall_metrics_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, marker in [("accuracy", "o"), ("f1", "s"), ("sensitivity", "^"), ("specificity", "v")]:
        values = [m[key] for m in val_overall_metrics_history]
        if any(v is None for v in values):
            continue
        ax.plot(epochs, values, label=key, marker=marker, markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{model_name} -- fold {fold_num} -- validation metrics over epochs")
    ax.legend()
    ax.grid(alpha=0.3)
    path = PLOTS_DIR / f"{model_name}_fold{fold_num}_val_metrics_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusion_matrices(model_name, fold_num, best_val_metrics) -> Path:
    buckets = ["overall", "India", "Italy"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, bucket in zip(axes, buckets):
        cm = best_val_metrics[bucket].get("confusion_matrix")
        n = best_val_metrics[bucket]["n"]
        if cm is None:
            ax.set_title(f"{bucket} (n={n})\nno data")
            ax.axis("off")
            continue
        cm = np.array(cm)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-anemic", "Anemic"])
        ax.set_yticklabels(["Non-anemic", "Anemic"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{bucket} (n={n})")
    fig.suptitle(f"{model_name} -- fold {fold_num} -- stratified confusion matrix (best val epoch)")
    fig.tight_layout()
    path = PLOTS_DIR / f"{model_name}_fold{fold_num}_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_roc_curves(model_name, fold_num, best_val_metrics) -> Path | None:
    buckets = ["overall", "India", "Italy"]
    overall_roc = best_val_metrics["overall"].get("roc_curve")
    if overall_roc is None:
        print(f"[{model_name} | fold {fold_num}] Skipping ROC plot -- val set had only one class present.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, bucket in zip(axes, buckets):
        roc = best_val_metrics[bucket].get("roc_curve")
        auc = best_val_metrics[bucket].get("auc")
        n = best_val_metrics[bucket]["n"]
        if roc is None:
            ax.set_title(f"{bucket} (n={n})\nonly one class present")
            ax.axis("off")
            continue
        ax.plot(roc["fpr"], roc["tpr"], label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{bucket} (n={n})")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model_name} -- fold {fold_num} -- stratified ROC curve (best val epoch)")
    fig.tight_layout()
    path = PLOTS_DIR / f"{model_name}_fold{fold_num}_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_kfold_test_summary(model_name: str, folds_df: pd.DataFrame) -> Path:
    """Cross-fold summary: overall test F1/Accuracy/AUC per fold plus the
    mean, and India/Italy test AUC per fold -- the two things this whole
    K-fold effort exists to report (a robust mean+/-std, and whether the
    confound gap survives across independently-trained folds)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    folds = folds_df["fold"].astype(str)
    width = 0.25
    x = np.arange(len(folds))
    for i, (col, label) in enumerate([("test_f1", "F1"), ("test_accuracy", "Accuracy"), ("test_auc", "AUC")]):
        if col in folds_df.columns:
            ax.bar(x + (i - 1) * width, folds_df[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {f}" for f in folds])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model_name}\ntest metrics per fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    for col, label, color in [("test_india_auc", "India AUC", "tab:orange"), ("test_italy_auc", "Italy AUC", "tab:blue")]:
        if col in folds_df.columns:
            ax2.plot(x, folds_df[col], marker="o", label=label, color=color)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"fold {f}" for f in folds])
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"{model_name}\ntest AUC by country, per fold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / f"{model_name}_kfold_test_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cross-fold summary plot to {path}")
    return path
