"""
EfficientNet-B0 (forniceal_palpebral) -- 5-Fold Cross-Validation pipeline.
Training engine.

Reuses build_efficientnet_b0() and compute_metrics() from the shared
../trainer_engine.py (pure/computational, no path dependency, safe to
import as-is) rather than duplicating them. Everything with a save path
or a changed training-loop behavior (gradient clipping, the LR scheduler,
plotting) is written fresh here, scoped to this pipeline's own outputs/ --
never classification/outputs/, so this experiment can never collide with
or overwrite the shared v2 pipeline's results.

Named cv_trainer_engine.py, not trainer_engine.py -- see cv_dataset.py's
docstring for why the naming collision with the shared file matters.

Locked hyperparameters (2026-07-30): extracted directly from Batch 1's
winning Optuna trial for efficientnet_b0_forniceal_palpebral_v2 (trial #9
of 12, best_val_f1=0.9333) -- see
classification/outputs/logs/efficientnet_b0_forniceal_palpebral_v2_study_summary.json.
Optuna is not used in this pipeline at all; these are fixed constants.
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

SHARED_SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # classification/scripts/
sys.path.insert(0, str(SHARED_SCRIPTS_DIR))
from trainer_engine import build_efficientnet_b0, compute_metrics  # noqa: E402 -- reuse, don't duplicate

from cv_dataset import get_fold_dataloaders, N_FOLDS, SEED  # noqa: E402

# --------------------------------------------------------------------------
# Locked hyperparameters -- from Batch 1 trial #9, not re-tuned here.
# --------------------------------------------------------------------------
MODEL_NAME = "efficientnet_b0_forniceal_palpebral_cv"
LEARNING_RATE = 0.0008200518402245837
WEIGHT_DECAY = 1.9634341572933354e-06
DROPOUT_RATE = 0.2

# --------------------------------------------------------------------------
# New settings for this pipeline (project author's confirmed roadmap).
# --------------------------------------------------------------------------
BATCH_SIZE = 32
MAX_EPOCHS = 150
GRAD_CLIP_MAX_NORM = 1.0  # not specified by the roadmap -- a standard, conservative default; flagged as my choice

# ReduceLROnPlateau: halves LR after 5 stagnant epochs, floors at 1e-6.
# EARLY_STOPPING_PATIENCE (15) is deliberately > SCHEDULER_PATIENCE (5) x2 --
# if it weren't, early stopping could fire before the scheduler ever gets a
# chance to reduce the LR and let training recover. Also my choice, flagged
# for the same reason as the grad-clip norm above.
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
MIN_LR = 1e-6
EARLY_STOPPING_PATIENCE = 15

NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"


# --------------------------------------------------------------------------
# Train / eval loops -- evaluate() reused as-is from the shared engine
# (pure function, no path dependency); train_one_epoch() rewritten here
# specifically to add gradient clipping, which the shared version doesn't
# have (and this pipeline deliberately doesn't add to the shared version,
# to avoid changing behavior for the other 18 already-run combos).
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


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_labels, all_probs, all_countries = [], [], []

    for images, labels, countries in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_countries.extend(countries)

    labels_arr = np.concatenate(all_labels)
    probs_arr = np.concatenate(all_probs)
    countries_arr = np.array(all_countries)
    metrics = compute_metrics(labels_arr, probs_arr, countries_arr)
    return total_loss / n_samples, metrics


# --------------------------------------------------------------------------
# One fold
# --------------------------------------------------------------------------
def run_fold(fold_num: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Trains EfficientNet-B0 (forniceal_palpebral) on one CV fold with a
    FRESH, randomly-initialized head (build_efficientnet_b0() constructs a
    new nn.Linear/backbone-projection every call -- no checkpoint is ever
    loaded here, by design, to strictly avoid the initialization-leakage
    concern raised and confirmed with the project author). Fixed
    hyperparameters throughout, no Optuna."""
    print(f"\n{'=' * 70}\nFold {fold_num}/{N_FOLDS} -- train={len(train_df)} val={len(val_df)}\n{'=' * 70}")

    torch.manual_seed(SEED + fold_num)  # reproducible but distinct per fold, not identically seeded across folds

    loaders = get_fold_dataloaders(train_df, val_df, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader, val_loader = loaders["train"], loaders["val"]

    train_labels = train_df["anemic_label"].to_numpy()
    n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    model = build_efficientnet_b0(DROPOUT_RATE).to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR
    )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / f"{MODEL_NAME}_fold{fold_num}_best.pth"

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
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Fold {fold_num}] New best val_f1={val_f1:.4f} -> saved {checkpoint_path.name}")

        print(
            f"[Fold {fold_num}] Epoch {epoch:>3}/{MAX_EPOCHS} - "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} lr={current_lr:.2e}"
        )

        scheduler.step(val_loss)  # ReduceLROnPlateau -- must come after computing val_loss this epoch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"[Fold {fold_num}] Early stopping at epoch {epoch}.")
                break

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    history = {
        "fold": fold_num,
        "n_epochs_run": epoch,
        "best_val_f1": best_val_f1,
        "best_val_metrics": best_val_metrics,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "lr_history": lr_history,
        "checkpoint_path": str(checkpoint_path),
    }
    history_path = LOGS_DIR / f"{MODEL_NAME}_fold{fold_num}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    _plot_loss_curve(fold_num, train_loss_history, val_loss_history)
    _plot_lr_curve(fold_num, lr_history)
    _plot_val_metrics_curve(fold_num, val_overall_metrics_history)
    _plot_confusion_matrices(fold_num, best_val_metrics)
    _plot_roc_curves(fold_num, best_val_metrics)

    print(f"[Fold {fold_num}] Done. Best val_f1={best_val_f1:.4f} after {epoch} epochs. Saved {history_path.name}")
    return history


# --------------------------------------------------------------------------
# Plots -- adapted from the shared engine's _plot_* functions, but writing
# to THIS pipeline's own PLOTS_DIR (the shared versions hardcode the
# shared classification/outputs/plots/ path, so they can't be reused
# as-is without silently writing into the wrong, shared location).
# --------------------------------------------------------------------------
def _plot_loss_curve(fold_num: int, train_loss_history: list, val_loss_history: list) -> Path:
    epochs = range(1, len(train_loss_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss_history, label="train loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss_history, label="val loss", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title(f"{MODEL_NAME} -- fold {fold_num} -- train vs val loss")
    ax.legend()
    ax.grid(alpha=0.3)
    path = PLOTS_DIR / f"{MODEL_NAME}_fold{fold_num}_loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_lr_curve(fold_num: int, lr_history: list) -> Path:
    """New plot type, not in the shared engine -- specifically to make
    ReduceLROnPlateau's behavior visible (when/how often it fired)."""
    epochs = range(1, len(lr_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, lr_history, marker="o", markersize=3, color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title(f"{MODEL_NAME} -- fold {fold_num} -- ReduceLROnPlateau schedule")
    ax.grid(alpha=0.3)
    path = PLOTS_DIR / f"{MODEL_NAME}_fold{fold_num}_lr_schedule.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_val_metrics_curve(fold_num: int, val_overall_metrics_history: list) -> Path:
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
    ax.set_title(f"{MODEL_NAME} -- fold {fold_num} -- validation metrics over epochs")
    ax.legend()
    ax.grid(alpha=0.3)
    path = PLOTS_DIR / f"{MODEL_NAME}_fold{fold_num}_val_metrics_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusion_matrices(fold_num: int, best_val_metrics: dict) -> Path:
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
    fig.suptitle(f"{MODEL_NAME} -- fold {fold_num} -- stratified confusion matrix")
    fig.tight_layout()
    path = PLOTS_DIR / f"{MODEL_NAME}_fold{fold_num}_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_roc_curves(fold_num: int, best_val_metrics: dict) -> Path | None:
    buckets = ["overall", "India", "Italy"]
    overall_roc = best_val_metrics["overall"].get("roc_curve")
    if overall_roc is None:
        print(f"[Fold {fold_num}] Skipping ROC plot -- val set had only one class present.")
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
    fig.suptitle(f"{MODEL_NAME} -- fold {fold_num} -- stratified ROC curve")
    fig.tight_layout()
    path = PLOTS_DIR / f"{MODEL_NAME}_fold{fold_num}_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
