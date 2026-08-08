"""
Shared Optuna training engine for the conjunctiva segmentation models.
Model-agnostic AND dataset-agnostic by design: it takes a model class/name
(or, for the pretrained-architecture sweep, a zero-arg build_model factory
-- see make_objective's docstring) and a dataset class, and runs the full
hyperparameter search against them. Each entry-point script (the 18 in
scripts/train_pretrained/, generated from models/segmentation/
pretrained_registry.py) just imports run_study() and passes in its own
model + dataset -- no shared file needs editing to pick which architecture
or dataset trains, which matters when execution happens on a remote
notebook (Kaggle) rather than this local environment.

(The original 3 hand-built architectures this engine trained -- Standard
U-Net, Attention U-Net, ResUNet, via train_standard_unet.py/
train_attention_unet.py/train_resunet.py and their _aligned.py
counterparts -- were removed once the pretrained-architecture sweep
superseded them; see CLAUDE.md Sec 2.1-2.5/3.5-3.6 for the historical
methodology/results record, intentionally kept even though the code and
Kaggle checkpoints/logs are gone.)

The loss function is also tuned by Optuna itself, not fixed per entry-point
script: each trial samples trial.suggest_categorical("loss_fn", [...]) from
LOSS_REGISTRY (currently "bce_dice" and "focal_tversky"), so a single study
directly compares both across trials -- see _save_outputs' per-loss-function
breakdown, and each loss gets its own best-checkpoint file.

Persists everything a Kaggle background run would otherwise lose when the
session ends: the best model's weights (outputs/checkpoints/), the full
per-trial metrics plus a best-trial summary (outputs/logs/), and a set of
plots covering every tracked metric (outputs/plots/) -- project author's
explicit request for the Kaggle output to include weights + plots + values
for every metric, downloadable as one zip (see the Kaggle notebook's
sync_outputs(), which now also collects outputs/plots/).
"""

import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    BATCH_SIZE,
    IMAGE_SIZE,
    SPLITS_CSV,
    AlignedConjunctivaSegmentationDataset,
    get_eval_transforms,
    get_train_transforms,
)
from segmentation_metrics import compute_batch_metrics, evaluate_final_test_set  # noqa: E402
from segmentation_plots import generate_all_plots  # noqa: E402

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
PLOTS_DIR = OUTPUTS_DIR / "plots"


# --------------------------------------------------------------------------
# Loss function
# --------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """Soft (differentiable) Dice loss computed from sigmoid probabilities
    directly on logits -- NOT the same as segmentation_metrics.compute_batch_metrics,
    which thresholds to a hard binary mask and can't be backpropagated through."""

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


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss (Abraham & Khan, 2018).

    Tversky index: TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    Focal Tversky:  FTL = (1 - TI) ** gamma

    Generalizes Dice (alpha == beta == 0.5 reduces to it) with independent
    false-positive/false-negative weights. beta > alpha (default 0.7 vs
    0.3) penalizes false negatives more than false positives -- directly
    countering the all-background collapse (every missed true-foreground
    pixel is a false negative). gamma > 1 (default 4/3, the original
    paper's value) down-weights already-easy samples so gradient
    concentrates on poorly-segmented ones -- useful on this project's
    202-patient aligned_raw set, whose foreground fraction is bimodal
    (median ~4%, but a small Italy-crop cluster runs ~75%), not uniformly
    sparse. Computed per-sample as a ratio, so unlike a global BCE
    pos_weight it adapts automatically to each image's own sparsity."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 4 / 3, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky_index = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        focal_tversky = (1 - tversky_index) ** self.gamma
        return focal_tversky.mean()


# Registry of loss functions available to the Optuna search (see
# make_objective) -- keyed by the name that shows up as trial.params["loss_fn"]
# and in the trials CSV as the "params_loss_fn" column.
LOSS_REGISTRY = {
    "bce_dice": BCEDiceLoss,
    "focal_tversky": FocalTverskyLoss,
}


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
    into a binary mask before Dice/IoU/Precision/Recall are computed (see
    segmentation_metrics.compute_batch_metrics) -- criterion (whichever
    LOSS_REGISTRY entry this trial sampled) itself still consumes raw
    logits directly, for numerical stability.

    Returns (avg_loss, avg_metrics) where avg_metrics is a dict with keys
    dice/iou/precision/recall -- each a correct sample-weighted mean across
    the full validation set (not a naive mean-of-batch-means), same
    weighting convention this function always used for loss/dice/iou."""
    model.eval()
    total_loss = 0.0
    total_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    n_samples = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = (torch.sigmoid(logits) > threshold).float()
        batch_metrics = compute_batch_metrics(preds, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        for key in total_metrics:
            total_metrics[key] += batch_metrics[key] * batch_size
        n_samples += batch_size

    avg_loss = total_loss / n_samples
    avg_metrics = {key: value / n_samples for key, value in total_metrics.items()}
    return avg_loss, avg_metrics


# --------------------------------------------------------------------------
# Per-model, one-time measurements (architecture properties, not
# hyperparameter-dependent -- computed once in run_study(), not per trial)
# --------------------------------------------------------------------------
def _build_fresh_model(model_cls, build_model):
    """Same construction branch objective() uses internally, factored out
    so run_study() can build an identical fresh instance for param-counting
    and latency measurement without duplicating the branching logic."""
    if build_model is not None:
        return build_model()
    return model_cls(in_channels=3, out_channels=1)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_inference_latency(model, image_size: int, device, n_warmup: int = 5, n_repeats: int = 20) -> float:
    """Average single-image (batch_size=1) forward-pass latency, in
    milliseconds. GPU calls are asynchronous, so an un-synchronized
    wall-clock measurement would mostly time kernel-launch overhead rather
    than actual compute -- torch.cuda.synchronize() before/after the timed
    loop is what makes this a real measurement. A handful of warmup passes
    are run first and excluded, since the first few CUDA calls include
    one-time kernel compilation/caching cost that isn't representative of
    steady-state inference speed."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_repeats):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / n_repeats * 1000.0


# --------------------------------------------------------------------------
# Optuna objective factory
# --------------------------------------------------------------------------
def make_objective(
    model_cls,
    model_name: str,
    dataset_cls=AlignedConjunctivaSegmentationDataset,
    build_model=None,
    image_size: int = None,
    tissue_type: str = None,
):
    """Builds an Optuna objective(trial) closure bound to a specific model
    class/name and dataset class, so the same engine can drive any
    segmentation architecture (in_channels=3, out_channels=1 -> raw-logits
    contract) against any dataset that returns (image, mask) pairs on that
    same contract. dataset_cls defaults to AlignedConjunctivaSegmentationDataset
    (raw-photo-aligned, CLAUDE.md Sec 1.4) -- the only segmentation dataset
    class remaining in this project since the original crop-based
    ConjunctivaSegmentationDataset was removed along with the 3 hand-built
    models that used it.

    Three optional parameters extend this for the 9-architecture pretrained
    sweep without needing per-architecture engine changes:
    - build_model: a zero-arg callable returning a ready nn.Module, used
      INSTEAD of `model_cls(in_channels=3, out_channels=1)` when given.
      This is what ARCHITECTURE_REGISTRY[name]["build_fn"] provides for the
      pretrained models (models/segmentation/pretrained_registry.py) --
      those builders need encoder_weights="imagenet"/from_pretrained(...)
      kwargs baked in, not a bare (in_channels, out_channels) contract.
    - image_size: resolution to resize to via get_train_transforms/
      get_eval_transforms. None (default) reproduces the original hardcoded
      256 behavior exactly. The 3 pretrained Transformer architectures need
      their own pretrained-checkpoint resolution (e.g. 224 or 512), same
      reasoning as classification's CNN-vs-transformer input_size split.
    - tissue_type: passed through to dataset_cls's constructor ONLY if that
      class actually accepts a tissue_type kwarg (checked via
      inspect.signature, not assumed).

    The closure also owns a `best_overall_dice` value that persists across
    every trial of the study (not just within one trial), so the checkpoint
    written to disk is always the single best-performing model seen across
    the whole search -- not just the last trial's own local best. It
    additionally tracks a best-dice-per-loss-function dict, so each loss
    function in LOSS_REGISTRY gets its own checkpoint too (see
    best_{model_name}_{loss_fn}.pth) -- needed for a side-by-side
    comparison, since the single "overall best" checkpoint alone would
    hide whichever loss function didn't happen to win outright."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"
    best_overall_dice = 0.0
    best_dice_per_loss_fn = {name: 0.0 for name in LOSS_REGISTRY}

    effective_image_size = image_size if image_size is not None else IMAGE_SIZE

    dataset_kwargs = {}
    if tissue_type is not None and "tissue_type" in inspect.signature(dataset_cls.__init__).parameters:
        dataset_kwargs["tissue_type"] = tissue_type

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_overall_dice, best_dice_per_loss_fn
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        loss_fn_name = trial.suggest_categorical("loss_fn", list(LOSS_REGISTRY.keys()))

        train_dataset = dataset_cls(
            split="train",
            splits_csv=SPLITS_CSV,
            transform=get_train_transforms(effective_image_size),
            **dataset_kwargs,
        )
        val_dataset = dataset_cls(
            split="val",
            splits_csv=SPLITS_CSV,
            transform=get_eval_transforms(effective_image_size),
            **dataset_kwargs,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        model = build_model().to(DEVICE) if build_model is not None else model_cls(in_channels=3, out_channels=1).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = LOSS_REGISTRY[loss_fn_name]()

        best_val_loss = float("inf")
        best_val_dice = 0.0
        best_val_metrics = {"iou": 0.0, "precision": 0.0, "recall": 0.0}
        epochs_without_improvement = 0
        epoch_history = []  # per-epoch record, kept only for the trial that ends up best (see below)

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)
            val_dice = val_metrics["dice"]

            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_metrics["iou"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                }
            )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_val_metrics = {"iou": val_metrics["iou"], "precision": val_metrics["precision"], "recall": val_metrics["recall"]}

                if val_dice > best_overall_dice:
                    best_overall_dice = val_dice
                    torch.save(model.state_dict(), checkpoint_path)
                    print(
                        f"[{model_name} | Trial {trial.number}] New best overall "
                        f"val_dice={val_dice:.4f} -> saved {checkpoint_path}"
                    )

                if val_dice > best_dice_per_loss_fn[loss_fn_name]:
                    best_dice_per_loss_fn[loss_fn_name] = val_dice
                    per_loss_checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}_{loss_fn_name}.pth"
                    torch.save(model.state_dict(), per_loss_checkpoint_path)
                    print(
                        f"[{model_name} | Trial {trial.number}] New best for loss_fn={loss_fn_name} "
                        f"val_dice={val_dice:.4f} -> saved {per_loss_checkpoint_path}"
                    )

            print(
                f"[{model_name} | Trial {trial.number} | loss_fn={loss_fn_name}] Epoch {epoch:>2}/{MAX_EPOCHS} - "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_dice={val_dice:.4f} val_iou={val_metrics['iou']:.4f} "
                f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"[{model_name} | Trial {trial.number} | loss_fn={loss_fn_name}] Early stopping "
                        f"at epoch {epoch} (no val_loss improvement for {EARLY_STOPPING_PATIENCE} epochs)."
                    )
                    break

        trial.set_user_attr("best_val_iou", best_val_metrics["iou"])
        trial.set_user_attr("best_val_precision", best_val_metrics["precision"])
        trial.set_user_attr("best_val_recall", best_val_metrics["recall"])
        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("loss_fn", loss_fn_name)
        # Kept on every trial (cheap -- MAX_EPOCHS x 7 floats), not just the
        # eventual best one, since Optuna decides which trial is "best" only
        # after every trial has already returned -- run_study() reads this
        # back from study.best_trial.user_attrs afterward, for the training
        # curve plots (project author's explicit request for plots of every
        # tracked metric, not just final numbers).
        trial.set_user_attr("epoch_history", epoch_history)
        return best_val_dice

    return objective


# --------------------------------------------------------------------------
# Study runner -- the single shared entry point every model-specific script calls
# --------------------------------------------------------------------------
def run_study(
    model_cls,
    model_name: str,
    dataset_cls=AlignedConjunctivaSegmentationDataset,
    n_trials: int = N_TRIALS,
    build_model=None,
    image_size: int = None,
    tissue_type: str = None,
) -> optuna.Study:
    print(f"Using device: {DEVICE}")
    print(f"Model: {model_name} ({model_cls.__name__ if model_cls is not None else 'pretrained build_model'})")
    print(f"Dataset: {dataset_cls.__name__}" + (f" (tissue_type={tissue_type})" if tissue_type else ""))
    print(f"Image size: {image_size if image_size is not None else IMAGE_SIZE}")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(
            model_cls,
            model_name,
            dataset_cls,
            build_model=build_model,
            image_size=image_size,
            tissue_type=tissue_type,
        ),
        n_trials=n_trials,
    )

    print("\n--- Optuna study complete ---")
    print(f"Model: {model_name}")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation Dice: {study.best_value:.4f}")
    print(f"Best validation IoU:  {study.best_trial.user_attrs['best_val_iou']:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    effective_image_size = image_size if image_size is not None else IMAGE_SIZE

    # Architecture-only properties (param count, inference latency) --
    # measured once on a fresh, untrained instance, since they don't depend
    # on which hyperparameters Optuna found, only on the architecture itself.
    print("\n--- Measuring architecture properties (params, inference latency) ---")
    probe_model = _build_fresh_model(model_cls, build_model).to(DEVICE)
    n_params = count_parameters(probe_model)
    latency_ms = measure_inference_latency(probe_model, effective_image_size, DEVICE)
    print(f"Parameters: {n_params:,}")
    print(f"Inference latency (batch_size=1, {effective_image_size}x{effective_image_size}): {latency_ms:.2f} ms")
    del probe_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Final TEST-set evaluation using the actual best checkpoint -- the test
    # split is never touched by the Optuna search or early stopping above,
    # so these are the numbers free of validation-set selection bias that
    # should be reported/compared, not the val-set numbers used to pick the
    # winning trial.
    test_metrics_summary = None
    test_df = None
    checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"
    if checkpoint_path.exists():
        print(f"\n--- Final test-set evaluation ({checkpoint_path.name}) ---")
        test_model = _build_fresh_model(model_cls, build_model).to(DEVICE)
        test_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        test_df = evaluate_final_test_set(
            test_model,
            dataset_cls,
            SPLITS_CSV,
            DEVICE,
            get_eval_transforms(effective_image_size),
            tissue_type=tissue_type,
        )
        del test_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        metric_cols = ["dice", "iou", "precision", "recall", "hd95"]
        overall = test_df[metric_cols].mean(numeric_only=True).to_dict()
        overall["hd95"] = float(np.nanmean(test_df["hd95"]))  # dice/iou/precision/recall never NaN; hd95 can be
        n_hd95_undefined = int(test_df["hd95"].isna().sum())
        per_country = test_df.groupby("country")[metric_cols].mean(numeric_only=True)
        per_country["hd95"] = test_df.groupby("country")["hd95"].apply(lambda s: float(np.nanmean(s)))

        test_metrics_summary = {
            "n_test_patients": len(test_df),
            "n_hd95_undefined_empty_mask": n_hd95_undefined,
            "overall": {k: float(v) for k, v in overall.items()},
            "by_country": per_country.to_dict(orient="index"),
        }
        print(f"Overall test Dice={overall['dice']:.4f} IoU={overall['iou']:.4f} "
              f"Precision={overall['precision']:.4f} Recall={overall['recall']:.4f} "
              f"HD95={overall['hd95']:.2f}px (undefined for {n_hd95_undefined}/{len(test_df)} patients)")
        print(per_country.round(4))
    else:
        print(f"\nNo checkpoint found at {checkpoint_path} -- skipping final test-set evaluation.")

    _save_outputs(study, model_name, n_params, latency_ms, effective_image_size, test_metrics_summary, test_df)
    return study


def _save_outputs(
    study: optuna.Study,
    model_name: str,
    n_params: int = None,
    latency_ms: float = None,
    image_size: int = None,
    test_metrics_summary: dict = None,
    test_df=None,
) -> None:
    """Persists everything needed to reconstruct this run's results after a
    Kaggle session ends: every trial's params/value/user_attrs as a CSV
    (including the "params_loss_fn" column from the loss_fn categorical
    hyperparameter, so bce_dice vs. focal_tversky trials are directly
    distinguishable), a compact JSON summary of the best trial overall, and
    -- if this study tuned loss_fn -- a per-loss-function comparison table
    (trial count, mean/max Dice per loss) both printed and saved, so the
    two losses can be compared without hand-filtering the trials CSV.

    n_params/latency_ms/image_size and test_metrics_summary/test_df are the
    additions for the broader thesis-defense metric set (project author's
    explicit request): architecture cost (params, inference latency) and
    final TEST-set metrics (Dice/IoU/Precision/Recall/HD95, overall and by
    country) go into the JSON summary; the full per-patient test_df is
    saved as its own CSV (`{model_name}_test_per_patient.csv`) since that
    per-patient granularity is exactly what compare_models_significance.py
    needs for a paired significance test against another model -- an
    aggregate-only summary can't support that."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    trials_csv_path = LOGS_DIR / f"{model_name}_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)

    summary = {
        "model_name": model_name,
        "n_trials_run": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_val_dice": study.best_value,
        "best_val_iou": study.best_trial.user_attrs["best_val_iou"],
        "best_val_precision": study.best_trial.user_attrs.get("best_val_precision"),
        "best_val_recall": study.best_trial.user_attrs.get("best_val_recall"),
        "best_params": study.best_params,
        "checkpoint_path": str(CHECKPOINTS_DIR / f"best_{model_name}.pth"),
        "n_params": n_params,
        "inference_latency_ms_batch1": latency_ms,
        "image_size": image_size,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if test_metrics_summary is not None:
        summary["test_set_metrics"] = test_metrics_summary

    if "params_loss_fn" in trials_df.columns:
        comparison = (
            trials_df.groupby("params_loss_fn")["value"]
            .agg(n_trials="count", mean_dice="mean", best_dice="max")
            .to_dict(orient="index")
        )
        summary["per_loss_fn_comparison"] = comparison
        summary["per_loss_fn_checkpoints"] = {
            loss_name: str(CHECKPOINTS_DIR / f"best_{model_name}_{loss_name}.pth")
            for loss_name in LOSS_REGISTRY
        }

        print("\n--- Per-loss-function comparison ---")
        print(trials_df.groupby("params_loss_fn")["value"].agg(["count", "mean", "max"]))

    summary_json_path = LOGS_DIR / f"{model_name}_study_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-trial metrics to {trials_csv_path}")
    print(f"Saved best-trial summary to {summary_json_path}")

    if test_df is not None:
        test_csv_path = LOGS_DIR / f"{model_name}_test_per_patient.csv"
        test_df.to_csv(test_csv_path, index=False)
        print(f"Saved per-patient test-set metrics to {test_csv_path}")

    # Plots -- one PNG per logical group of metrics (project author's
    # explicit request: Kaggle output should include plots of every
    # metric, not just raw values). epoch_history comes from the BEST
    # trial only (set_user_attr'd by every trial in make_objective, read
    # back here for whichever trial Optuna picked as the winner).
    epoch_history_raw = study.best_trial.user_attrs.get("epoch_history")
    epoch_history_df = pd.DataFrame(epoch_history_raw) if epoch_history_raw else None
    written_plots = generate_all_plots(
        PLOTS_DIR, model_name, epoch_history_df, trials_df, test_metrics_summary, test_df
    )
    if written_plots:
        print(f"Saved {len(written_plots)} plots to {PLOTS_DIR}:")
        for p in written_plots:
            print(f"  {p.name}")
