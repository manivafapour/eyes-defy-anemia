"""
Partial fine-tuning: unfreeze the FULL last block (features[7][1], stage 7
block 2, 2.24M params) of EfficientNet-B3, continuing training from the
already-converged, frozen-backbone checkpoint (Output/version1/checkpoints/
best_efficientnet_b3_forniceal_palpebral_new_way.pth, val F1=0.8966)
rather than starting from a fresh head.

Why the FULL last block, not a smaller sub-piece (unlike CoAtNet-3's
attention-sub-layer-only choice): discussed and decided in chat before any
code was written. EfficientNet-B3's last MBConv block splits into
expand+BN (889K), depthwise+BN (25K), squeeze-excite (445K, no norm layer
at all), and project+BN (886K) -- but only the full block legitimately
contains BatchNorm, which is the actual point of testing this specific
architecture (see below). A narrower choice like SE-only would dodge the
one thing this model was picked to test.

THE REAL DIFFERENCE FROM ConvNeXt-Base/CoAtNet-3, verified by inspection:
EfficientNet-B3 has BatchNorm everywhere (78 BatchNorm2d modules in the
whole model, 3 of them inside this one target block alone) -- neither
prior architecture had any BatchNorm in their fine-tune target region.
BatchNorm's running_mean/running_var are BUFFERS, not parameters --
requires_grad=False does NOT stop them from updating during .train() mode.
Left unhandled, every "frozen" BatchNorm layer elsewhere in the network
would silently drift its running statistics based on this project's small
per-epoch batches, contaminating the "same frozen backbone, only the
target block changed" premise the whole before/after comparison depends
on. Fixed via finetune_common.freeze_batchnorm_outside(), called every
epoch right after model.train() -- this file's own train_one_epoch()
override exists specifically to call it (same "small justified
duplication over parameterizing the shared version" pattern already used
for the other two engines' gradient-clipping addition).

Honest capacity caveat, flagged rather than glossed over: EfficientNet-B3's
whole backbone is only 10.7M params, the smallest of the three architectures
tried in this program -- so 2.24M is a *larger proportional* share (~21%)
of its backbone than either previous experiment was of theirs (ConvNeXt-Base
~9.6%, CoAtNet-3 ~5.8%), even though it's the smallest in absolute terms.
Given both previous experiments already showed signs of too much capacity
for this dataset at smaller proportional shares, this one is expected to
struggle at least as much -- the point of running it anyway is the
BatchNorm-handling validation as much as the fine-tuning outcome itself.

Train loop core logic and plotting are shared via finetune_common.py;
train_one_epoch is a thin local override (adds the BatchNorm-freezing call
that the shared version doesn't need for the other two architectures).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

FINE_TUNE_DIR = Path(__file__).resolve().parent  # classification/new_way/Fine_tune/
NEW_WAY_DIR = FINE_TUNE_DIR.parent  # classification/new_way/
CLASSIFICATION_DIR = NEW_WAY_DIR.parent  # classification/
DATAPREPAREPIPELINE_DIR = CLASSIFICATION_DIR / "datapreparepipeline"

sys.path.insert(0, str(DATAPREPAREPIPELINE_DIR))
sys.path.insert(0, str(NEW_WAY_DIR))
sys.path.insert(0, str(FINE_TUNE_DIR))

from trainer_engine import DEVICE, evaluate  # noqa: E402 -- reuse, don't duplicate
from dataset import TissueClassificationDataset, get_eval_transforms, get_train_transforms  # noqa: E402
from balanced_dataset import BalancedTissueClassificationDataset  # noqa: E402
import finetune_common as fc  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_NAME = "efficientnet_b3_forniceal_palpebral_new_way_finetune_block_v2"  # v2: lower LR after v1
# showed real training instability (train_loss spiking to 13-17 in several epochs, unlike either of
# the other two engines in this programme) -- distinct name, v1's outputs kept, not overwritten.
TISSUE_TYPE = "forniceal_palpebral"
IMAGE_SIZE = 256  # EfficientNet-B3 is CNN-family, resolution-flexible -- same 256 as ConvNeXt-Base

SOURCE_CHECKPOINT = NEW_WAY_DIR / "Output" / "version1" / "checkpoints" / "best_efficientnet_b3_forniceal_palpebral_new_way.pth"

# Original best trial (Output/version1/logs/efficientnet_b3_forniceal_palpebral_new_way_study_summary.json)
# -- read directly from that file, not hand-copied, at generation time by the entry-point script.
ORIGINAL_LEARNING_RATE = 0.006358358856676255
ORIGINAL_WEIGHT_DECAY = 0.000133112160807369
ORIGINAL_DROPOUT_RATE = 0.5

HEAD_LR_FACTOR = 0.2  # v2: reduced 5x from 1.0 (the original best-trial LR, unchanged from the other
# two engines) after v1's train_loss spiked erratically (13-17, vs. a normal ~0.4-1.1 range) --
# likely too aggressive for a block whose BatchNorm affine params and depthwise/expand/project
# convs are all adapting simultaneously, a failure mode neither ConvNeXt-Base nor CoAtNet-3 hit
# since neither had BatchNorm in their fine-tune target at all. Chosen to match ConvNeXt-Base's
# own v1 factor (0.2) for consistency, not independently tuned further.
BLOCK_LR_FACTOR = 0.1  # unchanged -- the newly-unfrozen block still gets 1/10 of the head's fine-tune
# LR (now itself 5x lower), so block_lr drops by the same 5x automatically.

BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

MAX_EPOCHS = 60
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 4
MIN_LR = 1e-7
EARLY_STOPPING_PATIENCE = 10  # > 2x SCHEDULER_PATIENCE
GRAD_CLIP_MAX_NORM = 1.0

OUTPUTS_DIR = FINE_TUNE_DIR / "Output"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"


# --------------------------------------------------------------------------
# Model: load the converged frozen-backbone checkpoint, then unfreeze only
# the last MBConv block (features[7][1]).
# --------------------------------------------------------------------------
def build_finetune_model(dropout_rate: float = ORIGINAL_DROPOUT_RATE, checkpoint_path: Path = SOURCE_CHECKPOINT) -> nn.Module:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found -- this fine-tune continues from the original "
            f"new_way Optuna-best checkpoint, which must exist first."
        )

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[0].p = dropout_rate  # overwrite the pre-existing fixed p=0.3 dropout (matches original build_efficientnet_b3)
    model.classifier[1] = nn.Linear(in_features, 1)  # fresh Linear -- requires_grad=True by construction

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)  # restores BOTH parameters and buffers (incl. BatchNorm running stats)

    for p in model.classifier[1].parameters():
        p.requires_grad = True
    for p in model.features[7][1].parameters():  # stage 7, block 2 (last block, 0-indexed [1])
        p.requires_grad = True

    return model


def _trainable_param_summary(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def get_loaders() -> dict:
    train_tf = get_train_transforms(IMAGE_SIZE)
    eval_tf = get_eval_transforms(IMAGE_SIZE)

    train_dataset = BalancedTissueClassificationDataset(tissue_type=TISSUE_TYPE, transform=train_tf)
    val_dataset = TissueClassificationDataset(split="val", tissue_type=TISSUE_TYPE, transform=eval_tf)
    test_dataset = TissueClassificationDataset(split="test", tissue_type=TISSUE_TYPE, transform=eval_tf)

    return {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
        "test": DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
    }


# --------------------------------------------------------------------------
# Train loop -- local override of finetune_common.train_one_epoch: same
# core loop (incl. gradient clipping), but also calls
# freeze_batchnorm_outside() every epoch right after model.train(), which
# the other two engines in this programme don't need (see module docstring).
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_max_norm: float, trainable_root: nn.Module) -> float:
    model.train()
    fc.freeze_batchnorm_outside(model, trainable_root)
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
# Main run
# --------------------------------------------------------------------------
def run_finetune(
    learning_rate: float = ORIGINAL_LEARNING_RATE,
    weight_decay: float = ORIGINAL_WEIGHT_DECAY,
    dropout_rate: float = ORIGINAL_DROPOUT_RATE,
) -> dict:
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Source checkpoint: {SOURCE_CHECKPOINT}")

    loaders = get_loaders()
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
    print(f"train n={len(train_loader.dataset)} val n={len(val_loader.dataset)} test n={len(test_loader.dataset)}")

    torch.manual_seed(SEED)

    model = build_finetune_model(dropout_rate, SOURCE_CHECKPOINT).to(DEVICE)
    trainable_root = model.features[7][1]
    param_summary = _trainable_param_summary(model)
    print(f"Trainable params: {param_summary['trainable_params']:,} / {param_summary['total_params']:,}")

    train_labels = train_loader.dataset.df["anemic_label"].to_numpy()
    n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("\n--- Baseline (frozen-backbone checkpoint, before fine-tuning) ---")
    baseline_val_loss, baseline_val_metrics = evaluate(model, val_loader, criterion, DEVICE)
    baseline_test_loss, baseline_test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    print(f"baseline val_f1={baseline_val_metrics['overall']['f1']:.4f} val_auc={baseline_val_metrics['overall'].get('auc')}")
    print(f"baseline test_f1={baseline_test_metrics['overall']['f1']:.4f} test_auc={baseline_test_metrics['overall'].get('auc')}")
    print(f"baseline India AUC={baseline_test_metrics['India'].get('auc')} Italy AUC={baseline_test_metrics['Italy'].get('auc')}")

    head_lr = learning_rate * HEAD_LR_FACTOR
    block_lr = head_lr * BLOCK_LR_FACTOR
    print(f"\nhead_lr={head_lr:.3e}  block_lr={block_lr:.3e}  weight_decay={weight_decay:.3e}")

    head_params = list(model.classifier[1].parameters())
    block_params = list(trainable_root.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr},
            {"params": block_params, "lr": block_lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR
    )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / f"{MODEL_NAME}_best.pth"

    best_val_f1 = -1.0
    best_val_metrics = None
    epochs_without_improvement = 0
    train_loss_history, val_loss_history = [], []
    head_lr_history, block_lr_history = [], []
    val_overall_metrics_history = []

    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, GRAD_CLIP_MAX_NORM, trainable_root)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        val_f1 = val_metrics["overall"]["f1"]

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        head_lr_history.append(optimizer.param_groups[0]["lr"])
        block_lr_history.append(optimizer.param_groups[1]["lr"])
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
            print(f"[{MODEL_NAME}] New best val_f1={val_f1:.4f} -> saved {checkpoint_path.name}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[{MODEL_NAME}] Epoch {epoch:>2}/{MAX_EPOCHS} - train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
            f"head_lr={optimizer.param_groups[0]['lr']:.2e} block_lr={optimizer.param_groups[1]['lr']:.2e}"
        )

        scheduler.step(val_f1)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"[{MODEL_NAME}] Early stopping at epoch {epoch} (no val_f1 improvement for {EARLY_STOPPING_PATIENCE} epochs).")
            break

    test_metrics = None
    if checkpoint_path.exists():
        print(f"\n--- Fine-tuned test-set evaluation ({checkpoint_path.name}) ---")
        test_model = build_finetune_model(dropout_rate, SOURCE_CHECKPOINT).to(DEVICE)
        test_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        _test_loss, test_metrics = evaluate(test_model, test_loader, criterion, DEVICE)
        print(f"fine-tuned test_f1={test_metrics['overall']['f1']:.4f} test_auc={test_metrics['overall'].get('auc')}")
        print(f"fine-tuned India AUC={test_metrics['India'].get('auc')} Italy AUC={test_metrics['Italy'].get('auc')}")
        del test_model
    else:
        print(f"[{MODEL_NAME}] No checkpoint saved -- skipping test-set evaluation.")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "model_name": MODEL_NAME,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "unfrozen": "features[7][1] (stage 7, block 2 of 2, full block incl. 3 BatchNorm layers) + classifier head",
        "trainable_params": param_summary["trainable_params"],
        "total_params": param_summary["total_params"],
        "hyperparameters": {
            "original_learning_rate": learning_rate,
            "head_lr": head_lr,
            "block_lr": block_lr,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
        },
        "n_epochs_run": epoch,
        "baseline_val_metrics": baseline_val_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "best_val_f1": best_val_f1,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "head_lr_history": head_lr_history,
        "block_lr_history": block_lr_history,
        "checkpoint_path": str(checkpoint_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    history_path = LOGS_DIR / f"{MODEL_NAME}_history.json"
    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved history to {history_path}")

    fc.print_before_after(baseline_test_metrics, test_metrics)
    fc.plot_loss_curve(MODEL_NAME, PLOTS_DIR, train_loss_history, val_loss_history)
    fc.plot_lr_curve(MODEL_NAME, PLOTS_DIR, {"head lr": head_lr_history, "block lr": block_lr_history})
    fc.plot_val_metrics_curve(MODEL_NAME, PLOTS_DIR, val_overall_metrics_history)
    if best_val_metrics is not None:
        fc.plot_confusion_matrices(MODEL_NAME, PLOTS_DIR, best_val_metrics)
        fc.plot_roc_curves(MODEL_NAME, PLOTS_DIR, best_val_metrics)
    if test_metrics is not None:
        fc.plot_before_after(MODEL_NAME, PLOTS_DIR, baseline_test_metrics, test_metrics)

    return result


if __name__ == "__main__":
    run_finetune()
