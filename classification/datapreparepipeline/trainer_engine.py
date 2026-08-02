"""
Phase 4 (Classification): shared Optuna training engine.

Independent of scripts/trainer_engine.py (the segmentation phase's engine)
-- separate metrics (accuracy/F1/AUC, not Dice/IoU), separate loss
(BCEWithLogitsLoss on a single-logit classification head, not a
segmentation loss), and a different model contract (ImageNet-pretrained
backbone + replaced head, not a from-scratch U-Net variant).

Architecture- and tissue-type-agnostic by design, mirroring the
segmentation phase's model-agnostic pattern: it takes an architecture name
(from ARCHITECTURE_REGISTRY) and a tissue_type ("palpebral" or
"forniceal_palpebral"), and the six thin entry-point scripts each just call
run_study() with their own choice of the two.

All backbones are frozen (only the replaced single-logit head is trained)
-- "transfer learning with frozen backbones" was requested explicitly, and
with only ~151 training patients, fine-tuning a full ImageNet backbone
end-to-end would be a serious overfitting risk for a first version.

Evaluation metrics are reported BOTH in aggregate AND stratified by country
(India vs Italy) as a first-class, always-computed output, not an
afterthought -- this project has an explicit, well-documented risk that a
model could achieve high aggregate accuracy purely by learning country-
correlated visual cues (lighting/camera differences between the India and
Italy acquisition sites) rather than actual conjunctival pallor, since
anemia prevalence is heavily confounded with country (CLAUDE.md Sec 0.5).
Aggregate accuracy alone cannot detect that failure mode; per-country
accuracy can (a model stuck near each country's majority-class rate despite
good aggregate accuracy is a red flag, not a success).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- this runs on Kaggle containers too, no display available
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import models

PIPELINE_DIR = Path(__file__).resolve().parent
MODULE_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))

from dataset import BATCH_SIZE, get_dataloaders  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
SEED = 42

MAX_EPOCHS = 250  # raised from 100, 2026-08-02 -- gives every trial more room to converge before
                  # early stopping (patience=7 below) decides it's done; early stopping is still
                  # the actual stopping mechanism for the vast majority of trials in practice
EARLY_STOPPING_PATIENCE = 7  # raised from 5 -- dropout=0.5 trials add per-epoch val_loss noise, so a
# too-tight patience risks stopping on a noisy bad epoch rather than genuine convergence, especially
# now that the epoch ceiling was deliberately raised to give slow-converging trials room to use it.
N_TRIALS = 12  # dropout_rate/lr/weight_decay search over a small (~151-patient) train set

OUTPUTS_DIR = MODULE_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

COUNTRIES = ["India", "Italy"]


# --------------------------------------------------------------------------
# Architecture registry: each entry builds an ImageNet-pretrained backbone
# with every parameter frozen except a freshly-initialized head, matching
# the raw-logit + BCEWithLogitsLoss contract used throughout this project
# (sigmoid applied externally only where a probability is actually needed
# -- at evaluation time here). The head is now Dropout(dropout_rate) ->
# Linear(in_features, 1) for every architecture, not a bare Linear --
# dropout_rate is an Optuna-tuned categorical (make_objective, below), so
# every architecture's regularization strength is actually searched rather
# than fixed or absent.
#
# Two of the nine (MobileNetV3-Small, EfficientNet-B0) ship a hardcoded
# p=0.2 Dropout inside their torchvision classifier already -- for those,
# the existing module's .p is overwritten with the trial's sampled value
# instead of adding a second stacked Dropout, so Optuna actually controls
# it rather than silently leaving it fixed. The other seven had no head
# dropout at all before this change (verified by inspecting each model's
# classifier/head/fc submodule directly, not assumed).
#
# input_size records each architecture's required input resolution for
# get_dataloaders() (make_objective, below). The six CNNs (three original +
# three new) are resolution-flexible via global/adaptive pooling before the
# classifier -- 256 was already proven to work (the original 6-combo run).
# The three transformers (Swin-T, ViT-B/16, ViT-L/16) are resolution-locked
# to their pretrained patch grid via position embeddings and need 224.
# --------------------------------------------------------------------------
def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def build_resnet18(dropout_rate: float) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))
    return model


def build_mobilenet_v3_small(dropout_rate: float) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.classifier[3].in_features
    model.classifier[2].p = dropout_rate  # overwrite the pre-existing fixed p=0.2 dropout
    model.classifier[3] = nn.Linear(in_features, 1)
    return model


def build_efficientnet_b0(dropout_rate: float) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.classifier[1].in_features
    model.classifier[0].p = dropout_rate  # overwrite the pre-existing fixed p=0.2 dropout
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def build_densenet121(dropout_rate: float) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))
    return model


def build_convnext_tiny(dropout_rate: float) -> nn.Module:
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))
    return model


def build_regnet_y_400mf(dropout_rate: float) -> nn.Module:
    model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))
    return model


def build_swin_t(dropout_rate: float) -> nn.Module:
    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.head.in_features
    model.head = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))
    return model


def build_vit_b_16(dropout_rate: float) -> nn.Module:
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))
    return model


def build_vit_l_16(dropout_rate: float) -> nn.Module:
    # IMAGENET1K_SWAG_LINEAR_V1, not the default IMAGENET1K_V1 -- verified the
    # default scores lower (79.7% top-1) than ViT-B/16 (81.1%) due to ViT-L
    # being notoriously undertrained on ImageNet-1K alone. SWAG_LINEAR_V1
    # (85.1% top-1) is a strictly better frozen feature extractor and still
    # uses 224x224 input, so it doesn't disturb the shared transformer
    # preprocessing size.
    model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
    _freeze_all(model)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, 1))
    return model


ARCHITECTURE_REGISTRY = {
    "resnet18": {"build_fn": build_resnet18, "input_size": 256},
    "mobilenet_v3_small": {"build_fn": build_mobilenet_v3_small, "input_size": 256},
    "efficientnet_b0": {"build_fn": build_efficientnet_b0, "input_size": 256},
    "densenet121": {"build_fn": build_densenet121, "input_size": 256},
    "convnext_tiny": {"build_fn": build_convnext_tiny, "input_size": 256},
    "regnet_y_400mf": {"build_fn": build_regnet_y_400mf, "input_size": 256},
    "swin_t": {"build_fn": build_swin_t, "input_size": 224},
    "vit_b_16": {"build_fn": build_vit_b_16, "input_size": 224},
    "vit_l_16": {"build_fn": build_vit_l_16, "input_size": 224},
}


# --------------------------------------------------------------------------
# Metrics: aggregate AND per-country (India vs Italy)
# --------------------------------------------------------------------------
def compute_metrics(labels: np.ndarray, probs: np.ndarray, countries: np.ndarray, threshold: float = 0.5) -> dict:
    """labels/probs/countries are aligned 1D arrays over one evaluation
    pass. Returns aggregate metrics plus a per-country breakdown -- the
    per-country lens is what actually exposes the India/Italy confound
    (CLAUDE.md Sec 0.5); aggregate accuracy alone cannot. Confusion matrix
    and ROC curve data (fpr/tpr) are computed for all three buckets
    (overall/India/Italy) uniformly -- cheap, and stratified confusion
    matrices AND ROC curves are both plotted per-bucket (see _save_outputs).

    v2 addition: sensitivity/specificity/balanced_accuracy, computed for
    all three buckets identically to every other metric here (not just
    "overall") -- every original key (accuracy/precision/recall/f1/auc/
    confusion_matrix/roc_curve) is preserved unchanged so the original
    6-combo results stay directly comparable to the v2 18-combo results,
    field for field."""
    preds = (probs > threshold).astype(float)

    def _safe_metrics(y_true, y_pred, y_prob):
        if len(y_true) == 0:
            return {"n": 0}
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        # specificity = TN / (TN + FP) -- sklearn has no direct function for
        # this, unlike recall/precision/f1, so it's computed straight from
        # the confusion matrix. None (not 0.0) when this slice has zero true
        # negatives, so it's never silently misread as "the model achieved
        # zero specificity" when the quantity is actually undefined.
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
        out = {
            "n": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": recall,
            "sensitivity": recall,  # alias -- sensitivity == recall for the positive (anemic) class
            "specificity": specificity,
            "balanced_accuracy": float((recall + specificity) / 2) if specificity is not None else None,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            # labels=[0,1] forces a full 2x2 matrix even if one class is
            # entirely absent from this slice (small per-country n makes
            # that a real possibility, not just a defensive formality).
            "confusion_matrix": cm.tolist(),
        }
        if len(set(y_true.tolist())) > 1:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, _thresholds = roc_curve(y_true, y_prob)
            out["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        else:
            out["auc"] = None  # undefined with only one class present in this slice
            out["roc_curve"] = None
        return out

    result = {"overall": _safe_metrics(labels, preds, probs)}
    for country in COUNTRIES:
        mask = countries == country
        result[country] = _safe_metrics(labels[mask], preds[mask], probs[mask])
    return result


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels, _countries in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    """Returns (avg_loss, metrics_dict). Predictions are sigmoid(logits) --
    criterion itself still consumes raw logits directly for numerical
    stability, same pattern as the segmentation phase (CLAUDE.md Sec 2.2)."""
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
# Optuna objective factory
# --------------------------------------------------------------------------
def make_objective(arch_name: str, tissue_type: str, model_name: str):
    """Closure bound to one (architecture, tissue_type) pair. Owns a
    best_overall_val_f1 value that persists across every trial in the
    study, so the checkpoint written to disk is always the single
    best-performing model seen across the whole search (same pattern as
    the segmentation engine's best_overall_dice)."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"
    best_overall_val_f1 = -1.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_overall_val_f1
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.2, 0.5])

        arch_config = ARCHITECTURE_REGISTRY[arch_name]
        loaders = get_dataloaders(
            tissue_type, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=arch_config["input_size"]
        )
        train_loader, val_loader = loaders["train"], loaders["val"]

        train_labels = train_loader.dataset.df["anemic_label"].to_numpy()
        n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

        model = arch_config["build_fn"](dropout_rate).to(DEVICE)
        # Only the replaced head has requires_grad=True (backbone frozen in
        # the builder above) -- filter so AdamW isn't handed frozen params.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        best_val_f1 = -1.0
        best_val_metrics = None
        epochs_without_improvement = 0
        train_loss_history = []
        val_loss_history = []
        val_overall_metrics_history = []  # per-epoch accuracy/f1/sensitivity/specificity, for the val-metrics-over-epochs plot

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)
            val_f1 = val_metrics["overall"]["f1"]
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
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

                if val_f1 > best_overall_val_f1:
                    best_overall_val_f1 = val_f1
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"[{model_name} | Trial {trial.number}] New best overall val_f1={val_f1:.4f} -> saved {checkpoint_path}")

            print(
                f"[{model_name} | Trial {trial.number}] Epoch {epoch:>2}/{MAX_EPOCHS} - "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_metrics['overall']['accuracy']:.4f} val_f1={val_f1:.4f} "
                f"India_acc={val_metrics['India']['accuracy'] if val_metrics['India']['n'] else float('nan'):.4f} "
                f"Italy_acc={val_metrics['Italy']['accuracy'] if val_metrics['Italy']['n'] else float('nan'):.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{model_name} | Trial {trial.number}] Early stopping at epoch {epoch}.")
                    break

        trial.set_user_attr("best_val_metrics", best_val_metrics)
        trial.set_user_attr("train_loss_history", train_loss_history)
        trial.set_user_attr("val_loss_history", val_loss_history)
        trial.set_user_attr("val_overall_metrics_history", val_overall_metrics_history)
        trial.set_user_attr("model_name", model_name)
        return best_val_f1

    return objective


# --------------------------------------------------------------------------
# Study runner -- the single shared entry point every entry-point script calls
# --------------------------------------------------------------------------
def run_study(arch_name: str, tissue_type: str, model_name: str, n_trials: int = N_TRIALS) -> optuna.Study:
    if arch_name not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"arch_name must be one of {list(ARCHITECTURE_REGISTRY)}, got {arch_name!r}")

    print(f"Using device: {DEVICE}")
    print(f"Architecture: {arch_name}")
    print(f"Tissue type: {tissue_type}")
    print(f"Model name: {model_name}")

    # n_startup_trials lowered from Optuna's default of 10, 2026-08-02 -- with N_TRIALS=12,
    # the default would spend 10/12 trials on pure random sampling before TPE's Bayesian
    # modeling ever engages. 5 leaves a real 7-trial informed-search budget instead of 2.
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(arch_name, tissue_type, model_name), n_trials=n_trials)

    print("\n--- Optuna study complete ---")
    print(f"Model: {model_name}")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation F1: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("Best trial's per-country validation breakdown:")
    print(json.dumps(study.best_trial.user_attrs["best_val_metrics"], indent=2))

    _save_outputs(study, model_name)
    return study


def _plot_loss_curve(model_name: str, trial_number: int, train_loss_history: list, val_loss_history: list) -> Path:
    epochs = range(1, len(train_loss_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss_history, label="train loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss_history, label="val loss", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title(f"{model_name} -- train vs val loss (best trial #{trial_number})")
    ax.legend()
    ax.grid(alpha=0.3)
    path = PLOTS_DIR / f"{model_name}_loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_roc_curves(model_name: str, trial_number: int, best_val_metrics: dict) -> Path:
    """Stratified ROC curve -- overall, India, Italy side by side, mirroring
    _plot_confusion_matrices below. v2 addition: the original engine only
    ever plotted the "overall" ROC curve (per-country fpr/tpr was computed
    but not plotted, since only the confusion matrix was explicitly
    requested stratified at the time). Per-country AUC gap is this
    project's primary confound-monitoring signal (CLAUDE.md Sec 0.5), so
    it's worth a citable figure of its own now, not just JSON data."""
    buckets = ["overall", "India", "Italy"]
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
    fig.suptitle(f"{model_name} -- stratified ROC curve (best trial #{trial_number})")
    fig.tight_layout()
    path = PLOTS_DIR / f"{model_name}_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_val_metrics_curve(model_name: str, trial_number: int, val_overall_metrics_history: list) -> Path:
    """v2 addition: validation accuracy/F1/sensitivity/specificity over
    epochs (overall bucket), all on one figure. Sensitivity+specificity
    together are the direct visual signal for this project's known
    "blanket-predict-anemic" collapse mode (sensitivity->1, specificity->0)
    -- more informative side by side than either metric alone."""
    epochs = range(1, len(val_overall_metrics_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, marker in [("accuracy", "o"), ("f1", "s"), ("sensitivity", "^"), ("specificity", "v")]:
        values = [m[key] for m in val_overall_metrics_history]
        if any(v is None for v in values):
            continue  # e.g. specificity undefined for an epoch with zero true negatives in this slice
        ax.plot(epochs, values, label=key, marker=marker, markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{model_name} -- validation metrics over epochs, overall (best trial #{trial_number})")
    ax.legend()
    ax.grid(alpha=0.3)
    path = PLOTS_DIR / f"{model_name}_val_metrics_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusion_matrices(model_name: str, trial_number: int, best_val_metrics: dict) -> Path:
    """Stratified confusion matrix -- overall, India, Italy side by side --
    the primary confound-monitoring artifact requested for this phase."""
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
    fig.suptitle(f"{model_name} -- stratified confusion matrix (best trial #{trial_number})")
    fig.tight_layout()
    path = PLOTS_DIR / f"{model_name}_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_outputs(study: optuna.Study, model_name: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    trials_csv_path = LOGS_DIR / f"{model_name}_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)

    best_trial = study.best_trial
    best_val_metrics = best_trial.user_attrs["best_val_metrics"]
    train_loss_history = best_trial.user_attrs["train_loss_history"]
    val_loss_history = best_trial.user_attrs["val_loss_history"]
    val_overall_metrics_history = best_trial.user_attrs.get("val_overall_metrics_history")

    # Plots are generated ONLY here, once, from the single best trial's data
    # -- never per-trial (explicit constraint: avoid plot clutter across an
    # entire Optuna search).
    plot_paths = {
        "loss_curve": str(_plot_loss_curve(model_name, best_trial.number, train_loss_history, val_loss_history)),
        "confusion_matrices": str(_plot_confusion_matrices(model_name, best_trial.number, best_val_metrics)),
    }

    if val_overall_metrics_history:
        plot_paths["val_metrics_curve"] = str(
            _plot_val_metrics_curve(model_name, best_trial.number, val_overall_metrics_history)
        )

    overall_roc = best_val_metrics["overall"].get("roc_curve")
    if overall_roc is not None:
        plot_paths["roc_curves"] = str(_plot_roc_curves(model_name, best_trial.number, best_val_metrics))
    else:
        print(f"[{model_name}] Skipping ROC curve plot -- best trial's val set had only one class present.")

    summary = {
        "model_name": model_name,
        "n_trials_run": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_val_f1": study.best_value,
        "best_val_metrics_by_country": best_val_metrics,
        "best_params": study.best_params,
        "checkpoint_path": str(CHECKPOINTS_DIR / f"best_{model_name}.pth"),
        "plot_paths": plot_paths,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_json_path = LOGS_DIR / f"{model_name}_study_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-trial metrics to {trials_csv_path}")
    print(f"Saved best-trial summary to {summary_json_path}")
    print(f"Saved plots (best trial only): {list(plot_paths.values())}")
