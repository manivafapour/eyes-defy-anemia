"""
Partial fine-tuning: unfreeze block 3 (the last block) of ConvNeXt-Large's
stage 4, continuing training from the already-converged, frozen-backbone
checkpoint (Output/version1/checkpoints/best_convnext_large_palpebral_new_way.pth,
val F1=0.9333) rather than starting from a fresh head.

Batch 2 of the fine-tune pilot programme (see classification/.project_memory/
13_finetune_pilot_programme.md) -- same "same absolute scale as the first 3
models" design decision, applied to the 3 next-best new_way combos.

Architecture (verified by inspection, weights=None, not assumed from the
ConvNeXt-Base precedent): ConvNeXt-Large uses the same 4-stage, [3,3,27,3]-
block topology as ConvNeXt-Base, just wider (192/384/768/1536 channels vs
Base's 128/256/512/1024). features[7] (stage 4) has 3 identical CNBlocks,
18,963,456 params each -- larger in absolute terms than ConvNeXt-Base's
8.448M/block purely because of this width difference, not a different
unfreezing choice; there is no smaller natural subdivision at the "last
block of last stage" level without breaking the block boundary, so this is
the honest same-pattern answer for this architecture. Zero BatchNorm
anywhere (verified), same as ConvNeXt-Base -- the frozen-but-still-
updating-running-stats gotcha does not apply here either.

Data/discriminative-LR/scheduler design is identical to finetune_engine.py
(ConvNeXt-Base) -- head continues at its original best-trial LR, block 3
gets a further 1/10 of that, scheduler/early-stopping track val_f1. Train
loop and plotting are shared via finetune_common.py.
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
MODEL_NAME = "convnext_large_palpebral_new_way_finetune_block3"
TISSUE_TYPE = "palpebral"
IMAGE_SIZE = 256

SOURCE_CHECKPOINT = NEW_WAY_DIR / "Output" / "version1" / "checkpoints" / "best_convnext_large_palpebral_new_way.pth"

# Original best trial (Output/version1/logs/convnext_large_palpebral_new_way_study_summary.json)
# -- read directly from that file, not hand-copied, at generation time by
# the entry-point script; the values below are this module's own fallback
# defaults, matching the same real numbers.
ORIGINAL_LEARNING_RATE = 0.00011137414908293505
ORIGINAL_WEIGHT_DECAY = 2.2059282146979313e-05
ORIGINAL_DROPOUT_RATE = 0.2

HEAD_LR_FACTOR = 1.0  # head continues at its original best-trial LR -- same corrected "v2" design
# the ConvNeXt-Base pilot converged on, applied directly from the start here (no reason to repeat
# the v1 mistake of starting too low).
BLOCK3_LR_FACTOR = 0.1  # block 3's LR is a further 1/10 of the head's fine-tune LR -- judgment call, flagged

BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

MAX_EPOCHS = 60  # shorter than the original 250-epoch budget -- refinement, not a from-scratch search
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
# Model: load the converged frozen-backbone checkpoint, then unfreeze
# only block 3 of stage 4 (features[7][2]).
# --------------------------------------------------------------------------
def build_finetune_model(dropout_rate: float = ORIGINAL_DROPOUT_RATE, checkpoint_path: Path = SOURCE_CHECKPOINT) -> nn.Module:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found -- this fine-tune continues from the original "
            f"new_way Optuna-best checkpoint, which must exist first."
        )

    model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Only classifier[2] (the replaced Dropout->Linear head) is "the head" --
    # classifier[0] is the original pretrained LayerNorm2d, which was frozen
    # during the original training too and must stay frozen here for
    # consistency (same bug class caught during ConvNeXt-Base's own
    # verification -- unfreezing the whole model.classifier Sequential would
    # silently also unfreeze that LayerNorm2d).
    for p in model.classifier[2].parameters():
        p.requires_grad = True
    for p in model.features[7][2].parameters():  # stage 4, block 3 (last block, 0-indexed [2])
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
    block3_lr = head_lr * BLOCK3_LR_FACTOR
    print(f"\nhead_lr={head_lr:.3e}  block3_lr={block3_lr:.3e}  weight_decay={weight_decay:.3e}")

    head_params = list(model.classifier[2].parameters())  # only the replaced Dropout->Linear head, not classifier[0]'s frozen LayerNorm2d
    block3_params = list(model.features[7][2].parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_lr},
            {"params": block3_params, "lr": block3_lr},
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
    head_lr_history, block3_lr_history = [], []
    val_overall_metrics_history = []

    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = fc.train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, GRAD_CLIP_MAX_NORM)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        val_f1 = val_metrics["overall"]["f1"]

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        head_lr_history.append(optimizer.param_groups[0]["lr"])
        block3_lr_history.append(optimizer.param_groups[1]["lr"])
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
            f"head_lr={optimizer.param_groups[0]['lr']:.2e} block3_lr={optimizer.param_groups[1]['lr']:.2e}"
        )

        scheduler.step(val_f1)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"[{MODEL_NAME}] Early stopping at epoch {epoch} (no val_f1 improvement for {EARLY_STOPPING_PATIENCE} epochs).")
            break

    # --- Final sealed-TEST evaluation, from a FRESH model loaded off the
    # saved best-val-F1 checkpoint. ---
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
        "unfrozen": "features[7][2] (stage 4, block 3 of 3) + classifier head",
        "trainable_params": param_summary["trainable_params"],
        "total_params": param_summary["total_params"],
        "hyperparameters": {
            "original_learning_rate": learning_rate,
            "head_lr": head_lr,
            "block3_lr": block3_lr,
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
        "block3_lr_history": block3_lr_history,
        "checkpoint_path": str(checkpoint_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    history_path = LOGS_DIR / f"{MODEL_NAME}_history.json"
    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved history to {history_path}")

    fc.print_before_after(baseline_test_metrics, test_metrics)
    fc.plot_loss_curve(MODEL_NAME, PLOTS_DIR, train_loss_history, val_loss_history)
    fc.plot_lr_curve(MODEL_NAME, PLOTS_DIR, {"head lr": head_lr_history, "block3 lr": block3_lr_history})
    fc.plot_val_metrics_curve(MODEL_NAME, PLOTS_DIR, val_overall_metrics_history)
    if best_val_metrics is not None:
        fc.plot_confusion_matrices(MODEL_NAME, PLOTS_DIR, best_val_metrics)
        fc.plot_roc_curves(MODEL_NAME, PLOTS_DIR, best_val_metrics)
    if test_metrics is not None:
        fc.plot_before_after(MODEL_NAME, PLOTS_DIR, baseline_test_metrics, test_metrics)

    return result


if __name__ == "__main__":
    run_finetune()
