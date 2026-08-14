"""
Partial fine-tuning: unfreeze the last MaxVitLayer of MaxViT-T's last stage
(blocks[3].layers[1], 8.96M params, includes 3 BatchNorm layers inside its
MBConv sub-part) + head, continuing training from the already-converged,
frozen-backbone checkpoint (Output/version1/checkpoints/
best_maxvit_t_palpebral_new_way.pth, val F1=0.8750) rather than starting
from a fresh head.

Batch 2 of the fine-tune pilot programme (see classification/.project_memory/
13_finetune_pilot_programme.md) -- same "same absolute scale as the first 3
models" design decision (~8-9M trainable), applied to the 3 next-best
new_way combos.

Architecture (verified by inspection, weights=None, not assumed): MaxViT-T
= stem + 4 stages (m.blocks, a ModuleList of 4 MaxVitBlock), stage sizes
294,952 / 1,114,064 / 11,165,456 / 17,530,688 params. The last stage
(blocks[3]) is itself a ModuleList of 2 MaxVitLayer objects (8,568,736 and
8,961,952 params) -- so, unlike ConvNeXt-Base/Large's uniform-block stages,
MaxViT-T's last stage naturally has a smaller final unit to pick, matching
the existing ~8-9M precedent directly rather than needing to go deeper.
**Chose the last MaxVitLayer as a whole** (blocks[3].layers[1]) rather than
splitting further into its own MBConv + 2x PartitionAttentionLayer
sub-parts (2.65M/0BN, 3.16M/0BN, 3.16M/0BN respectively, verified) --
picking the whole layer keeps this a direct "last unit of last stage"
choice, structurally parallel to what ConvNeXt-Base/Large and CoAtNet-3
already did, rather than a differently-scoped experiment.

BatchNorm check (verified by inspection, not assumed): 34 BatchNorm2d
modules total in the full model, all inside each stage's MBConv sub-parts
-- the target MaxVitLayer's own MBConv sub-part has 3 of them. Unlike
ConvNeXt-Base/Large/CoAtNet-3 (zero BatchNorm in their fine-tune targets),
this needs the same freeze_batchnorm_outside() fix already built for
EfficientNet-B3 -- this file has its own local train_one_epoch() override
that calls it, same "small justified duplication" pattern already used
there.

Data/discriminative-LR/scheduler design is identical to the other engines
in this programme -- head continues at its original best-trial LR, the
unfrozen layer gets a further 1/10 of that, scheduler/early-stopping track
val_f1. IMAGE_SIZE=224 matches this architecture's registered input size in
datapreparepipeline/trainer_engine.py's ARCHITECTURE_REGISTRY (same as
CoAtNet-3's engine, unlike the CNN engines' 256).
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
MODEL_NAME = "maxvit_t_palpebral_new_way_finetune_lastlayer"
TISSUE_TYPE = "palpebral"
IMAGE_SIZE = 224  # MaxViT-T is resolution-locked to its pretrained patch grid, NOT 256 like the CNNs

SOURCE_CHECKPOINT = NEW_WAY_DIR / "Output" / "version1" / "checkpoints" / "best_maxvit_t_palpebral_new_way.pth"

# Original best trial (Output/version1/logs/maxvit_t_palpebral_new_way_study_summary.json)
# -- read directly from that file, not hand-copied, at generation time by the entry-point script.
ORIGINAL_LEARNING_RATE = 0.0008179499475211679
ORIGINAL_WEIGHT_DECAY = 3.752055855124284e-05
ORIGINAL_DROPOUT_RATE = 0.2

HEAD_LR_FACTOR = 1.0  # head continues at its original best-trial LR -- same corrected "v2" design used
# by every other engine in this programme.
LASTLAYER_LR_FACTOR = 0.1  # the newly-unfrozen last layer gets 1/10 of the head's fine-tune LR --
# same ratio principle as every other engine, judgment call, flagged.

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
# the last MaxVitLayer of the last stage (blocks[3].layers[1]).
# --------------------------------------------------------------------------
def build_finetune_model(dropout_rate: float = ORIGINAL_DROPOUT_RATE, checkpoint_path: Path = SOURCE_CHECKPOINT) -> nn.Module:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found -- this fine-tune continues from the original "
            f"new_way Optuna-best checkpoint, which must exist first."
        )

    model = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[5].in_features
    model.classifier[5] = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Only classifier[5] (the replaced final Linear) is "the head" --
    # classifier[0:5] (AdaptiveAvgPool2d, Flatten, LayerNorm, the pretrained
    # 512->512 Linear+Tanh bottleneck) stay frozen, same as the original
    # frozen-backbone training (datapreparepipeline/trainer_engine.py's
    # build_maxvit_t() only ever replaced classifier[5]).
    for p in model.classifier[5].parameters():
        p.requires_grad = True
    for p in model.blocks[3].layers[1].parameters():  # last stage, last MaxVitLayer
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
# freeze_batchnorm_outside() every epoch right after model.train(), needed
# here (unlike ConvNeXt-Base/Large/CoAtNet-3) because the target
# MaxVitLayer's MBConv sub-part has real BatchNorm.
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
    trainable_root = model.blocks[3].layers[1]
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
    lastlayer_lr = head_lr * LASTLAYER_LR_FACTOR
    print(f"\nhead_lr={head_lr:.3e}  lastlayer_lr={lastlayer_lr:.3e}  weight_decay={weight_decay:.3e}")

    head_params = list(model.classifier[5].parameters())
    lastlayer_params = list(trainable_root.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr},
            {"params": lastlayer_params, "lr": lastlayer_lr},
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
    head_lr_history, lastlayer_lr_history = [], []
    val_overall_metrics_history = []

    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, GRAD_CLIP_MAX_NORM, trainable_root)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        val_f1 = val_metrics["overall"]["f1"]

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        head_lr_history.append(optimizer.param_groups[0]["lr"])
        lastlayer_lr_history.append(optimizer.param_groups[1]["lr"])
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
            f"head_lr={optimizer.param_groups[0]['lr']:.2e} lastlayer_lr={optimizer.param_groups[1]['lr']:.2e}"
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
        "unfrozen": "blocks[3].layers[1] (last stage, last MaxVitLayer, incl. 3 BatchNorm layers) + classifier[5]",
        "trainable_params": param_summary["trainable_params"],
        "total_params": param_summary["total_params"],
        "hyperparameters": {
            "original_learning_rate": learning_rate,
            "head_lr": head_lr,
            "lastlayer_lr": lastlayer_lr,
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
        "lastlayer_lr_history": lastlayer_lr_history,
        "checkpoint_path": str(checkpoint_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    history_path = LOGS_DIR / f"{MODEL_NAME}_history.json"
    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved history to {history_path}")

    fc.print_before_after(baseline_test_metrics, test_metrics)
    fc.plot_loss_curve(MODEL_NAME, PLOTS_DIR, train_loss_history, val_loss_history)
    fc.plot_lr_curve(MODEL_NAME, PLOTS_DIR, {"head lr": head_lr_history, "last layer lr": lastlayer_lr_history})
    fc.plot_val_metrics_curve(MODEL_NAME, PLOTS_DIR, val_overall_metrics_history)
    if best_val_metrics is not None:
        fc.plot_confusion_matrices(MODEL_NAME, PLOTS_DIR, best_val_metrics)
        fc.plot_roc_curves(MODEL_NAME, PLOTS_DIR, best_val_metrics)
    if test_metrics is not None:
        fc.plot_before_after(MODEL_NAME, PLOTS_DIR, baseline_test_metrics, test_metrics)

    return result


if __name__ == "__main__":
    run_finetune()
