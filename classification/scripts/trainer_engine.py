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

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

SCRIPTS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import BATCH_SIZE, get_dataloaders  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
SEED = 42

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
N_TRIALS = 12  # loss/lr/weight_decay search over a small (~151-patient) train set

OUTPUTS_DIR = MODULE_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"

COUNTRIES = ["India", "Italy"]


# --------------------------------------------------------------------------
# Architecture registry: each entry builds an ImageNet-pretrained backbone
# with every parameter frozen except a freshly-initialized single-logit
# head, matching the raw-logit + BCEWithLogitsLoss contract used throughout
# this project (sigmoid applied externally only where a probability is
# actually needed -- at evaluation time here).
# --------------------------------------------------------------------------
def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def build_resnet18() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def build_mobilenet_v3_small() -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 1)
    return model


def build_efficientnet_b0() -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    _freeze_all(model)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


ARCHITECTURE_REGISTRY = {
    "resnet18": build_resnet18,
    "mobilenet_v3_small": build_mobilenet_v3_small,
    "efficientnet_b0": build_efficientnet_b0,
}


# --------------------------------------------------------------------------
# Metrics: aggregate AND per-country (India vs Italy)
# --------------------------------------------------------------------------
def compute_metrics(labels: np.ndarray, probs: np.ndarray, countries: np.ndarray, threshold: float = 0.5) -> dict:
    """labels/probs/countries are aligned 1D arrays over one evaluation
    pass. Returns aggregate metrics plus a per-country breakdown -- the
    per-country lens is what actually exposes the India/Italy confound
    (CLAUDE.md Sec 0.5); aggregate accuracy alone cannot."""
    preds = (probs > threshold).astype(float)

    def _safe_metrics(y_true, y_pred, y_prob):
        if len(y_true) == 0:
            return {"n": 0}
        out = {
            "n": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(set(y_true.tolist())) > 1:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            out["auc"] = None  # undefined with only one class present in this slice
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

        loaders = get_dataloaders(tissue_type, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        train_loader, val_loader = loaders["train"], loaders["val"]

        train_labels = train_loader.dataset.df["anemic_label"].to_numpy()
        n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

        model = ARCHITECTURE_REGISTRY[arch_name]().to(DEVICE)
        # Only the replaced head has requires_grad=True (backbone frozen in
        # the builder above) -- filter so AdamW isn't handed frozen params.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        best_val_f1 = -1.0
        best_val_metrics = None
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)
            val_f1 = val_metrics["overall"]["f1"]

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

    sampler = optuna.samplers.TPESampler(seed=SEED)
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


def _save_outputs(study: optuna.Study, model_name: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    trials_csv_path = LOGS_DIR / f"{model_name}_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)

    summary = {
        "model_name": model_name,
        "n_trials_run": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_val_f1": study.best_value,
        "best_val_metrics_by_country": study.best_trial.user_attrs["best_val_metrics"],
        "best_params": study.best_params,
        "checkpoint_path": str(CHECKPOINTS_DIR / f"best_{model_name}.pth"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_json_path = LOGS_DIR / f"{model_name}_study_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-trial metrics to {trials_csv_path}")
    print(f"Saved best-trial summary to {summary_json_path}")
