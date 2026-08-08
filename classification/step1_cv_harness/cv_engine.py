"""
Step 1 -- Measurement Harness: per-fold training and out-of-fold prediction.

Deliberately minimal compared with the live trainer_engine.py: no Optuna (the
hyperparameters are locked to each combo's own winning trial), no checkpoint
writing. Unlike the original design, this DOES now track per-epoch loss
history per fold (added after the first CNN run made clear that visual,
per-fold diagnostics -- loss curves, confusion matrices, ROC curves -- were
wanted alongside the pooled statistics). The history is cheap (a couple of
Python floats appended per epoch); saving/plotting it lives in
run_cv_harness.py / cv_plots.py, not here.

The model built here is bit-identical to the one v2_clean trained -- same
frozen ImageNet backbone, same Dropout -> Linear head, same BCEWithLogitsLoss
with pos_weight, same AdamW -- because a baseline that measured a different
model would not be a baseline.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_config import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    NUM_WORKERS,
    PIPELINE_DIR,
    fit_seed,
)
from cv_data import make_fold_loaders  # noqa: E402

sys.path.insert(0, str(PIPELINE_DIR))
from trainer_engine import ARCHITECTURE_REGISTRY  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mutable_state(model: nn.Module) -> dict:
    """Snapshot everything that can actually change during a training epoch:
    the trainable head parameters, PLUS all buffers.

    The buffers matter and are easy to miss. requires_grad=False stops the
    optimizer from updating a BatchNorm layer's weight/bias, but it does NOT
    freeze its running_mean/running_var -- those keep updating on every
    forward pass in train() mode, in 5 of the 6 CNNs here. Restoring only the
    head would therefore pair the best epoch's head with the LAST epoch's
    normalization statistics, which is not the model that achieved the best
    inner-validation loss.

    Frozen backbone weights are deliberately excluded: they are byte-identical
    at every epoch, so copying ~30-110MB per epoch would buy nothing.
    """
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    buffers = {name for name, _ in model.named_buffers()}
    keep = trainable | buffers
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k in keep}


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels, _countries, _pids in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def predict(model, loader, criterion, device) -> dict:
    """Returns per-patient probabilities plus the mean loss. sigmoid is applied
    here only; the criterion still consumes raw logits, matching the
    numerical-stability convention used across this project."""
    model.eval()
    total_loss, n = 0.0, 0
    probs, labels_out, countries_out, pids_out = [], [], [], []
    for images, labels, countries, pids in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels_out.append(labels.cpu().numpy())
        countries_out.extend(list(countries))
        pids_out.extend(list(pids))
    return {
        "loss": total_loss / max(n, 1),
        "probs": np.concatenate(probs) if probs else np.array([]),
        "labels": np.concatenate(labels_out) if labels_out else np.array([]),
        "countries": np.array(countries_out),
        "patient_ids": np.array(pids_out),
    }


def run_fold(pool_df, fold: dict, combo: dict, base_seed: int, verbose: bool = True) -> dict:
    """Trains one fold and returns its out-of-fold predictions.

    Early stopping watches the INNER validation loss and the best inner-loss
    epoch's head weights are restored before predicting on the outer fold, so
    the outer fold contributes a genuinely held-out prediction -- it is never
    consulted, directly or indirectly, during training or epoch selection.
    """
    _seed_everything(fit_seed(base_seed, fold["repeat"], fold["fold"]))

    arch_config = ARCHITECTURE_REGISTRY[combo["arch_name"]]
    loaders = make_fold_loaders(
        pool_df,
        fold,
        tissue_type=combo["tissue_type"],
        image_size=arch_config["input_size"],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # pos_weight from this fold's inner-training portion only. Computing it
    # over the whole pool would leak the outer fold's class balance into the
    # loss -- small, but it is exactly the kind of leak Step 1 exists to avoid.
    inner_labels = loaders["inner_train"].dataset.df["anemic_label"].to_numpy()
    n_pos = float(inner_labels.sum())
    n_neg = float(len(inner_labels) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32).to(DEVICE)

    model = arch_config["build_fn"](combo["dropout_rate"]).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=combo["learning_rate"], weight_decay=combo["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_inner_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    train_loss_history = []
    inner_val_loss_history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, loaders["inner_train"], optimizer, criterion, DEVICE)
        inner = predict(model, loaders["inner_val"], criterion, DEVICE)
        train_loss_history.append(float(train_loss))
        inner_val_loss_history.append(float(inner["loss"]))
        if inner["loss"] < best_inner_loss:
            best_inner_loss = inner["loss"]
            best_state = _mutable_state(model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    outer = predict(model, loaders["outer_val"], criterion, DEVICE)
    if verbose:
        print(
            f"    r{fold['repeat']}f{fold['fold']}: stopped epoch {epoch} "
            f"(best inner epoch {best_epoch}, inner_loss={best_inner_loss:.4f}, "
            f"train_loss={train_loss:.4f}) -> {len(outer['patient_ids'])} held-out predictions",
            flush=True,
        )

    del model, optimizer, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "repeat": fold["repeat"],
        "fold": fold["fold"],
        "n_epochs_run": epoch,
        "best_inner_epoch": best_epoch,
        "best_inner_loss": float(best_inner_loss),
        "train_loss_history": train_loss_history,
        "inner_val_loss_history": inner_val_loss_history,
        "patient_ids": outer["patient_ids"],
        "probs": outer["probs"],
        "labels": outer["labels"],
        "countries": outer["countries"],
    }
