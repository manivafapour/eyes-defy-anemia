"""
Shared Optuna training engine for the palpebral conjunctiva segmentation
models. Model-agnostic AND dataset-agnostic by design: it takes a model
class/name and a dataset class, and runs the full 5-trial hyperparameter
search against them. The per-model entry-point scripts (train_standard_
unet.py, train_attention_unet.py, train_resunet.py, and their _aligned.py
counterparts) each just import run_study() and pass in their own model +
dataset -- no shared file needs editing to pick which architecture or
dataset trains, which matters when execution happens on a remote notebook
(Kaggle) rather than this local environment.

Persists everything a Kaggle background run would otherwise lose when the
session ends: the best model's weights (outputs/checkpoints/) and the full
per-trial metrics plus a best-trial summary (outputs/logs/).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    BATCH_SIZE,
    SPLITS_CSV,
    ConjunctivaSegmentationDataset,
    get_eval_transforms,
    get_train_transforms,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
SEED = 42

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
N_TRIALS = 5

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"


# --------------------------------------------------------------------------
# Loss function
# --------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """Soft (differentiable) Dice loss computed from sigmoid probabilities
    directly on logits -- NOT the same as compute_dice_iou below, which
    thresholds to a hard binary mask and can't be backpropagated through."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.eps) / (probs.sum(dim=1) + targets.sum(dim=1) + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCEWithLogitsLoss + soft Dice loss, averaged with bce_weight.

    Plain per-pixel BCE lets a model minimize its loss by predicting
    all-background whenever the true foreground is a tiny fraction of the
    image -- with the raw-photo-aligned dataset, per-patient foreground can
    be well under 1% of the 256x256 canvas (CLAUDE.md Sec 1.4), so BCE's
    per-pixel average barely moves for getting that tiny region wrong,
    and training can collapse to an all-zero prediction (val_dice pinned
    at 0.0000 even as val_loss keeps decreasing). Dice loss is a ratio, not
    a per-pixel average, so it stays scale-invariant to how small the true
    foreground is and keeps penalizing an all-background prediction
    heavily regardless."""

    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=eps)
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + (1 - self.bce_weight) * self.dice(logits, targets)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_dice_iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    """Mean Dice and IoU over a batch of binary [B, 1, H, W] tensors."""
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    dice = (2 * intersection + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.5):
    """Validation pass. Predictions are sigmoid(logits) thresholded at 0.5
    into a binary mask before Dice/IoU are computed -- criterion (BCEDiceLoss)
    itself still consumes raw logits directly, for numerical stability."""
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    n_samples = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = (torch.sigmoid(logits) > threshold).float()
        dice, iou = compute_dice_iou(preds, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        n_samples += batch_size

    return total_loss / n_samples, total_dice / n_samples, total_iou / n_samples


# --------------------------------------------------------------------------
# Optuna objective factory
# --------------------------------------------------------------------------
def make_objective(model_cls, model_name: str, dataset_cls=ConjunctivaSegmentationDataset):
    """Builds an Optuna objective(trial) closure bound to a specific model
    class/name and dataset class, so the same engine can drive any
    segmentation architecture (in_channels=3, out_channels=1 -> raw-logits
    contract) against any dataset that returns (image, mask) pairs on that
    same contract -- e.g. ConjunctivaSegmentationDataset (crop-based) or
    AlignedConjunctivaSegmentationDataset (raw-photo-aligned, CLAUDE.md
    Sec 1.4). dataset_cls defaults to the original crop-based dataset so
    existing callers that don't pass it are unaffected.

    The closure also owns a `best_overall_dice` value that persists across
    every trial of the study (not just within one trial), so the checkpoint
    written to disk is always the single best-performing model seen across
    the whole search -- not just the last trial's own local best."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"
    best_overall_dice = 0.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_overall_dice
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        train_dataset = dataset_cls(
            split="train", splits_csv=SPLITS_CSV, transform=get_train_transforms()
        )
        val_dataset = dataset_cls(
            split="val", splits_csv=SPLITS_CSV, transform=get_eval_transforms()
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        model = model_cls(in_channels=3, out_channels=1).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = BCEDiceLoss()

        best_val_loss = float("inf")
        best_val_dice = 0.0
        best_val_iou = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, DEVICE)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_val_iou = val_iou

                if val_dice > best_overall_dice:
                    best_overall_dice = val_dice
                    torch.save(model.state_dict(), checkpoint_path)
                    print(
                        f"[{model_name} | Trial {trial.number}] New best overall "
                        f"val_dice={val_dice:.4f} -> saved {checkpoint_path}"
                    )

            print(
                f"[{model_name} | Trial {trial.number}] Epoch {epoch:>2}/{MAX_EPOCHS} - "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"[{model_name} | Trial {trial.number}] Early stopping at epoch {epoch} "
                        f"(no val_loss improvement for {EARLY_STOPPING_PATIENCE} epochs)."
                    )
                    break

        trial.set_user_attr("best_val_iou", best_val_iou)
        trial.set_user_attr("model_name", model_name)
        return best_val_dice

    return objective


# --------------------------------------------------------------------------
# Study runner -- the single shared entry point every model-specific script calls
# --------------------------------------------------------------------------
def run_study(
    model_cls,
    model_name: str,
    dataset_cls=ConjunctivaSegmentationDataset,
    n_trials: int = N_TRIALS,
) -> optuna.Study:
    print(f"Using device: {DEVICE}")
    print(f"Model: {model_name} ({model_cls.__name__})")
    print(f"Dataset: {dataset_cls.__name__}")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(model_cls, model_name, dataset_cls), n_trials=n_trials)

    print("\n--- Optuna study complete ---")
    print(f"Model: {model_name}")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation Dice: {study.best_value:.4f}")
    print(f"Best validation IoU:  {study.best_trial.user_attrs['best_val_iou']:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    _save_outputs(study, model_name)
    return study


def _save_outputs(study: optuna.Study, model_name: str) -> None:
    """Persists everything needed to reconstruct this run's results after a
    Kaggle session ends: every trial's params/value/user_attrs as a CSV, and
    a compact JSON summary of the best trial (including where its checkpoint
    was written by make_objective, so both files can be cross-referenced)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    trials_csv_path = LOGS_DIR / f"{model_name}_trials.csv"
    study.trials_dataframe().to_csv(trials_csv_path, index=False)

    summary = {
        "model_name": model_name,
        "n_trials_run": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_val_dice": study.best_value,
        "best_val_iou": study.best_trial.user_attrs["best_val_iou"],
        "best_params": study.best_params,
        "checkpoint_path": str(CHECKPOINTS_DIR / f"best_{model_name}.pth"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_json_path = LOGS_DIR / f"{model_name}_study_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-trial metrics to {trials_csv_path}")
    print(f"Saved best-trial summary to {summary_json_path}")
