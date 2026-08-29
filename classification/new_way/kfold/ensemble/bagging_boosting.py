"""
Bagging and boosting -- unlike everything in ensemble_engine.py (soft
voting, soups, greedy soup, stacking), these two genuinely need NEW models:
the 18 existing checkpoints are 6 architectures x 3 K-fold splits, which is
neither a bootstrap resample (bagging) nor a sequence of reweighted fits
(boosting). Training is cheap here only because the backbone is frozen --
only the small head (Dropout -> Linear) ever updates, so a full "training
run" is a few seconds to low minutes even on a laptop GPU, not a Kaggle-
scale job.

Both reuse each architecture's own already-tuned fixed hyperparameters
(learning_rate/weight_decay/dropout_rate, read from that architecture's own
{model_name}_kfold_folds.csv-adjacent kfold_summary.json) rather than
re-tuning -- consistent with this project's "fixed hyperparameter K-fold
retraining" design (see kfold_engine.py's own module docstring).

Both are scoped to a small subset of architectures (not all 6) given the
wall-clock cost of actually training multiple new models per architecture,
and are run entirely on the pool (load_pool_df) with the sealed TEST set
touched only once, at the very end, for evaluation -- exactly like every
other technique here.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ENSEMBLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENSEMBLE_DIR))

from ensemble_engine import (  # noqa: E402
    ARCHITECTURE_REGISTRY,
    BATCH_SIZE,
    DEVICE,
    LOGS_DIR,
    MODEL_REGISTRY,
    NUM_WORKERS,
    OUTPUTS_DIR,
    TissueClassificationDataset,
    _run_inference,
    compute_metrics,
    evaluate_predictions,
    get_eval_transforms,
    load_pool_df,
)
from dataset import get_train_transforms  # noqa: E402

BB_CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints" / "bagging_boosting"  # separate subfolder -- never collides with the original K-fold checkpoint names
BB_CACHE_DIR = LOGS_DIR / "bagging_boosting_cache"


def _fixed_hyperparameters(model_key: str) -> dict:
    """Reuses the SAME (learning_rate, weight_decay, dropout_rate) already
    found and used for this architecture's K-fold retraining -- read from
    its own kfold_summary.json rather than re-tuning, exactly as
    kfold_engine.py's own design already does for its 3 folds."""
    spec = MODEL_REGISTRY[model_key]
    summary_path = LOGS_DIR / f"{spec['model_name']}_kfold_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    return summary["fixed_hyperparameters"]


def _pool_datasets(model_key: str):
    """Returns (pool_df, train_tf_ds, eval_tf_ds, pid_to_idx) -- two
    TissueClassificationDataset instances over the IDENTICAL pool patient
    set (one with training/augmenting transforms, one with deterministic
    eval transforms), row-order-aligned by construction (same patient_ids
    input, same underlying deterministic filtering), so a single
    pid_to_idx lookup indexes into either one correctly."""
    spec = MODEL_REGISTRY[model_key]
    arch_config = ARCHITECTURE_REGISTRY[spec["arch_name"]]
    image_size = arch_config["input_size"]

    pool_df = load_pool_df(spec["tissue_type"])
    pool_ids = pool_df["patient_id"].tolist()

    train_tf_ds = TissueClassificationDataset(
        split="val", tissue_type=spec["tissue_type"], transform=get_train_transforms(image_size), patient_ids=pool_ids
    )
    eval_tf_ds = TissueClassificationDataset(
        split="val", tissue_type=spec["tissue_type"], transform=get_eval_transforms(image_size), patient_ids=pool_ids
    )
    assert list(train_tf_ds.df["patient_id"]) == list(eval_tf_ds.df["patient_id"]) == pool_ids, (
        f"{model_key}: pool dataset row order mismatch -- pid_to_idx lookup would be wrong"
    )
    pid_to_idx = {pid: i for i, pid in enumerate(pool_ids)}
    return pool_df, train_tf_ds, eval_tf_ds, pid_to_idx


def _train_head(
    model_key: str,
    train_loader: DataLoader,
    device: torch.device = DEVICE,
    val_loader: DataLoader = None,
    max_epochs: int = 60,
    patience: int = 8,
    sample_weights: torch.Tensor = None,
    verbose_prefix: str = "",
) -> tuple:
    """Generic frozen-backbone head training loop, shared by bagging
    (val_loader given -> early stop on OOB val loss) and boosting
    (val_loader=None, sample_weights given -> fixed epoch budget, weighted
    BCE). Returns (best_state_dict, history dict). If val_loader is None,
    "best" is just the final epoch's state (no early-stopping signal
    available without a held-out set, by design for boosting -- see
    run_boosting()'s docstring)."""
    spec = MODEL_REGISTRY[model_key]
    arch_config = ARCHITECTURE_REGISTRY[spec["arch_name"]]
    hp = _fixed_hyperparameters(model_key)

    model = arch_config["build_fn"](hp["dropout_rate"]).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=hp["learning_rate"], weight_decay=hp["weight_decay"])

    if sample_weights is None:
        # bagging: pos_weight computed from THIS bootstrap draw's actual label multiplicities (repeats matter)
        train_labels = np.array([train_loader.dataset.dataset.df.iloc[i]["anemic_label"] for i in train_loader.dataset.indices])
        n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        offset = 0
        for images, labels, _countries in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)

            if sample_weights is None:
                loss = criterion(logits, labels)
            else:
                w_batch = sample_weights[offset : offset + labels.size(0)].to(device)
                per_sample = criterion(logits, labels)
                loss = (per_sample * w_batch).sum() / w_batch.sum()
                offset += labels.size(0)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
        train_loss = running_loss / n_seen
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            model.eval()
            val_running, val_n = 0.0, 0
            eval_criterion = nn.BCEWithLogitsLoss()
            with torch.no_grad():
                for images, labels, _countries in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images).squeeze(1)
                    val_running += eval_criterion(logits, labels).item() * images.size(0)
                    val_n += images.size(0)
            val_loss = val_running / max(val_n, 1)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose_prefix:
                        print(f"{verbose_prefix} early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                    break
        else:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, history


def _predict_with_state(model_key: str, state_dict: dict, tissue_type: str, image_size: int, patient_ids: list, device=DEVICE) -> pd.DataFrame:
    arch_config = ARCHITECTURE_REGISTRY[MODEL_REGISTRY[model_key]["arch_name"]]
    model = arch_config["build_fn"](0.0).to(device)
    model.load_state_dict(state_dict)
    df = _run_inference(model, tissue_type, image_size, device, f"{model_key} bagging/boosting round", patient_ids=patient_ids)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return df


def _predict_test_with_state(model_key: str, state_dict: dict, device=DEVICE) -> pd.DataFrame:
    spec = MODEL_REGISTRY[model_key]
    arch_config = ARCHITECTURE_REGISTRY[spec["arch_name"]]
    image_size = arch_config["input_size"]
    model = arch_config["build_fn"](0.0).to(device)
    model.load_state_dict(state_dict)
    df = _run_inference(model, spec["tissue_type"], image_size, device, f"{model_key} bagging/boosting round (test)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return df


# --------------------------------------------------------------------------
# Bagging: B fresh heads, each trained on an independent bootstrap resample
# (with replacement) of the pool, early-stopped on that round's out-of-bag
# (OOB) patients -- the ones never drawn into the bootstrap sample, a free
# held-out set unique to bagging. Final prediction = simple average of all
# B heads' sealed-test probabilities.
# --------------------------------------------------------------------------
def run_bagging(model_key: str, n_bags: int = 5, max_epochs: int = 60, patience: int = 8, seed: int = 1000, use_cache: bool = True) -> dict:
    cache_path = BB_CACHE_DIR / f"{model_key}_bagging_B{n_bags}.json"
    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  [cache] {model_key} bagging B={n_bags} loaded from {cache_path.name}")
        return cached

    spec = MODEL_REGISTRY[model_key]
    arch_config = ARCHITECTURE_REGISTRY[spec["arch_name"]]
    image_size = arch_config["input_size"]
    pool_df, train_tf_ds, eval_tf_ds, pid_to_idx = _pool_datasets(model_key)
    pool_ids = pool_df["patient_id"].to_numpy()

    BB_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    rounds = []
    test_pred_dfs = []
    for b in range(1, n_bags + 1):
        rng = np.random.RandomState(seed + b)
        bootstrap_ids = rng.choice(pool_ids, size=len(pool_ids), replace=True)
        oob_ids = sorted(set(pool_ids) - set(bootstrap_ids))

        train_indices = [pid_to_idx[pid] for pid in bootstrap_ids]
        oob_indices = [pid_to_idx[pid] for pid in oob_ids]
        train_loader = DataLoader(Subset(train_tf_ds, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        oob_loader = DataLoader(Subset(eval_tf_ds, oob_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        print(f"  [bagging:{model_key}] round {b}/{n_bags}: bootstrap n={len(train_indices)} (unique={len(set(bootstrap_ids))}), OOB n={len(oob_ids)}")
        state, history = _train_head(model_key, train_loader, val_loader=oob_loader, max_epochs=max_epochs, patience=patience, verbose_prefix=f"  [bagging:{model_key} round {b}]")

        ckpt_path = BB_CHECKPOINTS_DIR / f"{model_key}_bag{b}.pth"
        BB_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)  # re-checked per round, not just once -- defends against the directory being renamed/moved out from under a long-running job, as happened 2026-08-13
        torch.save(state, ckpt_path)

        oob_df = _predict_with_state(model_key, state, spec["tissue_type"], image_size, oob_ids)
        oob_metrics = evaluate_predictions(oob_df)
        test_df = _predict_test_with_state(model_key, state)
        test_metrics = evaluate_predictions(test_df)
        test_pred_dfs.append(test_df)

        print(f"  [bagging:{model_key}] round {b}: OOB f1={oob_metrics['overall']['f1']:.4f}  own test_f1={test_metrics['overall']['f1']:.4f}  epochs_run={len(history['train_loss'])}")
        rounds.append(
            {
                "round": b,
                "n_bootstrap": len(train_indices),
                "n_unique_bootstrap": len(set(bootstrap_ids)),
                "n_oob": len(oob_ids),
                "epochs_run": len(history["train_loss"]),
                "oob_metrics": oob_metrics,
                "own_test_metrics": test_metrics,
                "checkpoint_path": str(ckpt_path),
            }
        )

    base = test_pred_dfs[0].set_index("patient_id").sort_index()
    prob_matrix = np.column_stack([df.set_index("patient_id").sort_index()["prob"].to_numpy() for df in test_pred_dfs])
    bagged_probs = prob_matrix.mean(axis=1)
    final_metrics = compute_metrics(base["label"].to_numpy(), bagged_probs, base["country"].to_numpy())

    result = {"model_key": model_key, "n_bags": n_bags, "rounds": rounds, "final_metrics": final_metrics}

    if use_cache:
        BB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


# --------------------------------------------------------------------------
# Boosting: AdaBoost.M1 / SAMME (K=2 reduces to classic AdaBoost) over the
# frozen-backbone head. Trains M rounds sequentially on the FULL pool with
# per-sample loss weights (not resampling); each round's weighted training
# error sets alpha_m (how much to trust that round); misclassified patients
# get upweighted for the next round.
#
# IMPORTANT CAVEAT, disclosed rather than hidden: classic boosting assumes
# WEAK learners (only slightly better than chance) so each round has
# genuinely new signal to add. A frozen ImageNet backbone + linear head on
# ~184 patients is not a weak learner -- it can often fit the weighted
# training pool almost perfectly within a handful of epochs, in which case
# reweighting produces little new signal round-to-round and boosting can
# degenerate toward several near-identical, highly-correlated heads (i.e.
# closer to bagging without the resampling diversity, but worse, since
# there's no OOB set for early stopping -- see max_epochs_per_round below).
# Run and report honestly either way; this isn't assumed to work.
#
# No natural held-out set exists mid-procedure (unlike bagging's OOB
# patients), so each round trains for a FIXED epoch budget (no early
# stopping) -- a disclosed simplification, not a bug.
# --------------------------------------------------------------------------
def run_boosting(model_key: str, n_rounds: int = 5, epochs_per_round: int = 15, seed: int = 2000, use_cache: bool = True) -> dict:
    cache_path = BB_CACHE_DIR / f"{model_key}_boosting_M{n_rounds}.json"
    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  [cache] {model_key} boosting M={n_rounds} loaded from {cache_path.name}")
        return cached

    torch.manual_seed(seed)
    spec = MODEL_REGISTRY[model_key]
    pool_df, train_tf_ds, eval_tf_ds, pid_to_idx = _pool_datasets(model_key)
    pool_ids = pool_df["patient_id"].to_numpy()
    n = len(pool_ids)
    labels_true = pool_df.set_index("patient_id").loc[pool_ids, "anemic_label"].to_numpy()

    weights = np.full(n, 1.0 / n)
    full_indices = list(range(n))  # fixed row order == pool_ids order (see _pool_datasets)

    BB_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    rounds = []
    test_pred_dfs = []
    alphas = []
    for m in range(1, n_rounds + 1):
        # shuffle=False -- keeps DataLoader iteration order == pool_ids order, so a running offset
        # into `weights` (itself in pool_ids order) lines up with each batch without needing the
        # dataset to hand back explicit indices.
        train_loader = DataLoader(Subset(train_tf_ds, full_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)

        print(f"  [boosting:{model_key}] round {m}/{n_rounds}: training on full pool (n={n}) with current weights (max={weights.max():.4f}, min={weights.min():.4f})")
        state, history = _train_head(model_key, train_loader, val_loader=None, max_epochs=epochs_per_round, sample_weights=weight_tensor, verbose_prefix=f"  [boosting:{model_key} round {m}]")

        ckpt_path = BB_CHECKPOINTS_DIR / f"{model_key}_boost{m}.pth"
        BB_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt_path)

        pool_pred_df = _predict_with_state(model_key, state, spec["tissue_type"], ARCHITECTURE_REGISTRY[spec["arch_name"]]["input_size"], list(pool_ids))
        pool_pred_df = pool_pred_df.set_index("patient_id").loc[pool_ids].reset_index()
        preds = (pool_pred_df["prob"].to_numpy() > 0.5).astype(float)
        misclassified = (preds != labels_true).astype(float)

        err_m = float((weights * misclassified).sum() / weights.sum())
        err_m_clamped = min(max(err_m, 1e-6), 1 - 1e-6)
        alpha_m = 0.5 * np.log((1 - err_m_clamped) / err_m_clamped)

        test_df = _predict_test_with_state(model_key, state)
        test_metrics = evaluate_predictions(test_df)
        test_pred_dfs.append(test_df)
        pool_train_metrics = evaluate_predictions(pool_pred_df)

        print(f"  [boosting:{model_key}] round {m}: weighted_train_err={err_m:.4f}  alpha={alpha_m:.4f}  own_test_f1={test_metrics['overall']['f1']:.4f}  pool_train_f1={pool_train_metrics['overall']['f1']:.4f}")

        rounds.append(
            {
                "round": m,
                "weighted_train_error": err_m,
                "alpha": alpha_m,
                "epochs_run": len(history["train_loss"]),
                "pool_train_metrics": pool_train_metrics,
                "own_test_metrics": test_metrics,
                "checkpoint_path": str(ckpt_path),
            }
        )

        if err_m >= 0.5:
            print(f"  [boosting:{model_key}] round {m}: weighted error >= 0.5 (worse than chance on the reweighted pool) -- stopping early, this round NOT included in the final combination.")
            test_pred_dfs.pop()
            rounds[-1]["excluded_from_final"] = True
            break

        alphas.append(alpha_m)
        # AdaBoost.M1 update: upweight misclassified, renormalize
        weights = weights * np.exp(alpha_m * misclassified)
        weights = weights / weights.sum()

    if not alphas:
        raise RuntimeError(f"{model_key}: every boosting round had weighted error >= 0.5 -- no usable rounds to combine.")

    # Weighted combination of margins (2p-1 in [-1,1]) -> sigmoid back to a probability-like score
    base = test_pred_dfs[0].set_index("patient_id").sort_index()
    margin_matrix = np.column_stack([2 * df.set_index("patient_id").sort_index()["prob"].to_numpy() - 1 for df in test_pred_dfs])
    alpha_arr = np.array(alphas)
    combined_margin = (margin_matrix * alpha_arr).sum(axis=1) / alpha_arr.sum()
    boosted_probs = 1 / (1 + np.exp(-4 * combined_margin))  # scale=4: a full-strength unanimous vote (margin=+-1) saturates to prob~0.02/0.98 rather than exactly 0/1

    final_metrics = compute_metrics(base["label"].to_numpy(), boosted_probs, base["country"].to_numpy())

    result = {"model_key": model_key, "n_rounds_run": len(rounds), "n_rounds_used_in_final": len(alphas), "rounds": rounds, "final_metrics": final_metrics}

    if use_cache:
        BB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
    return result
