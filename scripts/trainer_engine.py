"""
Shared Optuna training engine for the palpebral conjunctiva segmentation
models. Model-agnostic by design: it takes a model class and a name and
runs the full 5-trial hyperparameter search against them. The per-model
entry-point scripts (train_standard_unet.py, train_attention_unet.py,
train_resunet.py) each just import run_study() and pass in their own model
-- no shared file needs editing to pick which architecture trains, which
matters when execution happens on a remote notebook (Kaggle) rather than
this local environment.
"""

import sys
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
    into a binary mask before Dice/IoU are computed -- BCEWithLogitsLoss
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
def make_objective(model_cls, model_name: str):
    """Builds an Optuna objective(trial) closure bound to a specific model
    class/name, so the same engine can drive any segmentation architecture
    that follows the (in_channels=3, out_channels=1) -> raw-logits contract."""

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        train_dataset = ConjunctivaSegmentationDataset(
            split="train", splits_csv=SPLITS_CSV, transform=get_train_transforms()
        )
        val_dataset = ConjunctivaSegmentationDataset(
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
        criterion = nn.BCEWithLogitsLoss()

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
def run_study(model_cls, model_name: str, n_trials: int = N_TRIALS) -> optuna.Study:
    print(f"Using device: {DEVICE}")
    print(f"Model: {model_name} ({model_cls.__name__})")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(model_cls, model_name), n_trials=n_trials)

    print("\n--- Optuna study complete ---")
    print(f"Model: {model_name}")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation Dice: {study.best_value:.4f}")
    print(f"Best validation IoU:  {study.best_trial.user_attrs['best_val_iou']:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study
