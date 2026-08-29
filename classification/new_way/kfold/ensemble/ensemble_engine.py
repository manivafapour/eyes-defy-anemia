"""
Post-hoc soft-voting ensembles over the already-trained new_way/kfold/ checkpoints.

No retraining. This loads saved per-fold checkpoints (Output/version2/checkpoints/
{model_name}_fold{n}_best.pth), runs inference on the sealed 33-patient TEST split,
and averages sigmoid probabilities across a chosen subset of checkpoints ("soft
voting"), then applies the same threshold=0.5 rule every single-model metric in
this project already uses (trainer_engine.compute_metrics).

Why this is valid: kfold_engine.py's TEST loader is never touched by the K-fold
split (build_folds() only partitions the pooled train+val patients) -- it always
reads TissueClassificationDataset(split="test", tissue_type=tissue_type), i.e. the
same 33 rows of classification/data/processed/splits.csv, filtered by that tissue
type's "ok" extraction status. Verified directly against splits.csv +
extraction_log.csv (not assumed): all 33 test patients have BOTH palpebral and
forniceal_palpebral crops marked "ok". So every checkpoint's test predictions --
including efficientnet_b3, the one architecture trained on the forniceal_palpebral
crop instead of palpebral -- line up 1:1 by patient_id. Predictions are merged on
patient_id explicitly (not positionally) so this holds regardless of any row-order
difference between the two tissue types' filtered dataframes.

Determinism / built-in correctness check: eval-mode inference here (frozen
backbone, dropout disabled, deterministic Resize+Normalize transform, no
shuffling) is identical to what kfold_engine.run_fold()'s own sealed-test
evaluation already did when each checkpoint was produced. So a freshly
recomputed single-checkpoint metric here should reproduce that checkpoint's
recorded row in {model_name}_kfold_folds.csv exactly -- verify_against_recorded()
below checks this for every baseline and flags a loud mismatch instead of
silently trusting a new code path.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ENSEMBLE_DIR = Path(__file__).resolve().parent  # classification/new_way/kfold/ensemble/
KFOLD_DIR = ENSEMBLE_DIR.parent  # classification/new_way/kfold/
NEW_WAY_DIR = KFOLD_DIR.parent  # classification/new_way/
CLASSIFICATION_DIR = NEW_WAY_DIR.parent  # classification/
DATAPREPAREPIPELINE_DIR = CLASSIFICATION_DIR / "datapreparepipeline"

sys.path.insert(0, str(DATAPREPAREPIPELINE_DIR))
sys.path.insert(0, str(NEW_WAY_DIR))  # balanced_dataset.py -- imported transitively by kfold_engine.py below
sys.path.insert(0, str(KFOLD_DIR))

from trainer_engine import ARCHITECTURE_REGISTRY, DEVICE, compute_metrics  # noqa: E402 -- reuse, don't duplicate
from dataset import EXTRACTION_LOG_CSV, SPLITS_CSV, TissueClassificationDataset, get_eval_transforms  # noqa: E402
from kfold_engine import build_folds, load_pool_df  # noqa: E402 -- reused so OOF fold assignment is byte-identical to what actually produced the checkpoints (same seed=42, same pool_df row order)

# --------------------------------------------------------------------------
# The 6 trained new_way/kfold/ combos (Output/version2/). Kept as an explicit
# small registry rather than parsed from filenames -- fragile to infer
# arch_name/tissue_type from a checkpoint's name alone (e.g. "convnext_base"
# vs "convnext_large" both start with "convnext_"), and this mirrors the
# same explicit-registry pattern already used for the 6 thin
# train_kfold_*.py entry points.
# --------------------------------------------------------------------------
MODEL_REGISTRY = {
    "coatnet_3": {"arch_name": "coatnet_3", "tissue_type": "palpebral", "model_name": "coatnet_3_palpebral_new_way_kfold3"},
    "convnext_base": {"arch_name": "convnext_base", "tissue_type": "palpebral", "model_name": "convnext_base_palpebral_new_way_kfold3"},
    "convnext_large": {"arch_name": "convnext_large", "tissue_type": "palpebral", "model_name": "convnext_large_palpebral_new_way_kfold3"},
    "efficientnet_b3": {"arch_name": "efficientnet_b3", "tissue_type": "forniceal_palpebral", "model_name": "efficientnet_b3_forniceal_palpebral_new_way_kfold3"},
    "maxvit_t": {"arch_name": "maxvit_t", "tissue_type": "palpebral", "model_name": "maxvit_t_palpebral_new_way_kfold3"},
    "regnet_y_16gf": {"arch_name": "regnet_y_16gf", "tissue_type": "palpebral", "model_name": "regnet_y_16gf_palpebral_new_way_kfold3"},
}
N_FOLDS = 3
BATCH_SIZE = 32
NUM_WORKERS = 0  # matches kfold_engine.py -- Windows-safe, sealed test set is only 33 patients anyway

OUTPUTS_DIR = NEW_WAY_DIR / "Output" / "version2"  # the exact dir kfold_engine.py writes to (its 18 fold checkpoints + per-fold logs live here); this ensemble suite writes ensemble_*/bagging_boosting_* alongside them, never into version1/ (the Optuna sweep)
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PRED_CACHE_DIR = LOGS_DIR / "ensemble_pred_cache"  # per-checkpoint patient-level probs, cached to disk

_PRED_CACHE: dict = {}  # in-memory cache for this process, keyed by (model_key, fold)


# --------------------------------------------------------------------------
# Sealed-test verification (run once, importable so callers can assert on it)
# --------------------------------------------------------------------------
def verify_shared_test_set() -> set:
    """Returns the set of 33 test patient_ids and raises if either tissue
    type's 'ok' filter would drop any of them -- the precondition this whole
    module depends on for merging predictions across tissue types by
    patient_id."""
    splits = pd.read_csv(SPLITS_CSV)
    log = pd.read_csv(EXTRACTION_LOG_CSV)
    test_ids = set(splits.loc[splits["split"] == "test", "patient_id"])
    for tissue_type in ["palpebral", "forniceal_palpebral"]:
        ok_ids = set(log.loc[log[f"{tissue_type}_status"] == "ok", "patient_id"])
        missing = test_ids - ok_ids
        if missing:
            raise RuntimeError(
                f"{len(missing)} sealed-test patient(s) have no usable '{tissue_type}' crop "
                f"({sorted(missing)}) -- cross-tissue-type ensembling would silently drop them "
                f"from any merge. Restrict ensembles to same-tissue-type checkpoints instead."
            )
    return test_ids


# --------------------------------------------------------------------------
# Per-checkpoint inference on the sealed test set
# --------------------------------------------------------------------------
def _checkpoint_path(model_key: str, fold: int) -> Path:
    spec = MODEL_REGISTRY[model_key]
    return CHECKPOINTS_DIR / f"{spec['model_name']}_fold{fold}_best.pth"


def _run_inference(
    model: torch.nn.Module, tissue_type: str, image_size: int, device: torch.device, error_tag: str, patient_ids: list = None
) -> pd.DataFrame:
    """Shared forward-pass loop for a single checkpoint, a souped
    (weight-averaged) model, or an OOF pass over one fold's own held-out
    val_ids -- everything downstream of "here is a loaded, eval-mode
    model" is identical either way. patient_ids=None (default) means the
    sealed TEST split; an explicit list restricts to exactly those patients
    instead (TissueClassificationDataset's patient_ids param overrides its
    split param when given -- split="val" here is just a label, ignored).
    Returns a DataFrame [patient_id, country, label, prob]."""
    model.eval()
    if patient_ids is None:
        dataset = TissueClassificationDataset(split="test", tissue_type=tissue_type, transform=get_eval_transforms(image_size))
    else:
        dataset = TissueClassificationDataset(
            split="val", tissue_type=tissue_type, transform=get_eval_transforms(image_size), patient_ids=patient_ids
        )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_labels, all_probs, all_countries = [], [], []
    with torch.no_grad():
        for images, labels, countries in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            all_countries.extend(countries)

    # shuffle=False -> loader order == dataset.df row order, so
    # patient_id can be read straight off the dataset's own dataframe.
    result_ids = dataset.df["patient_id"].tolist()
    labels_arr = np.concatenate(all_labels)
    probs_arr = np.concatenate(all_probs)
    assert len(result_ids) == len(labels_arr) == len(probs_arr) == len(all_countries), (
        f"{error_tag}: patient_id/label/prob/country length mismatch -- "
        f"loader must not be shuffled for the patient_id alignment above to hold."
    )

    return pd.DataFrame(
        {
            "patient_id": result_ids,
            "country": all_countries,
            "label": labels_arr,
            "prob": probs_arr,
        }
    )


def predict_checkpoint(model_key: str, fold: int, device: torch.device = DEVICE, use_cache: bool = True) -> pd.DataFrame:
    """Returns a DataFrame [patient_id, country, label, prob] -- one row per
    sealed-test patient -- for a single (architecture, fold) checkpoint.
    Cached both in-memory and on disk (PRED_CACHE_DIR), since the same
    checkpoint is typically reused across several named ensembles."""
    cache_key = (model_key, fold)
    if cache_key in _PRED_CACHE:
        return _PRED_CACHE[cache_key].copy()

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"model_key must be one of {list(MODEL_REGISTRY)}, got {model_key!r}")
    if not (1 <= fold <= N_FOLDS):
        raise ValueError(f"fold must be in 1..{N_FOLDS}, got {fold}")

    spec = MODEL_REGISTRY[model_key]
    cache_csv = PRED_CACHE_DIR / f"{spec['model_name']}_fold{fold}_test_probs.csv"
    if use_cache and cache_csv.exists():
        df = pd.read_csv(cache_csv)
        _PRED_CACHE[cache_key] = df
        return df.copy()

    checkpoint_path = _checkpoint_path(model_key, fold)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint for {model_key} fold {fold}: {checkpoint_path}. "
            f"Was classification/new_way/kfold/train_kfold_{model_key}_*.py ever run for this fold?"
        )

    arch_config = ARCHITECTURE_REGISTRY[spec["arch_name"]]
    image_size = arch_config["input_size"]

    # dropout_rate is irrelevant here -- Dropout has no learnable parameters,
    # model.eval() disables it regardless of the value used to construct it.
    model = arch_config["build_fn"](0.0).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)  # checkpoints are saved fp16 (kfold_engine._half_state_dict); copy_ upcasts automatically into this fp32 model
    model.load_state_dict(state_dict)

    df = _run_inference(model, spec["tissue_type"], image_size, device, f"{model_key} fold {fold}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _PRED_CACHE[cache_key] = df
    if use_cache:
        PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df.copy()


# --------------------------------------------------------------------------
# Model soups: average CHECKPOINT WEIGHTS across a subset of one
# architecture's folds into a single merged model, instead of averaging
# forward-pass probabilities across separately-run models (soft voting,
# above). Only valid WITHIN one architecture -- different architectures
# have differently-shaped weight tensors and can't be averaged at all.
#
# Why this is a reasonable thing to try for THIS project specifically: all
# 3 folds of a given architecture share the exact same frozen, ImageNet-
# pretrained backbone (never trained, so its plain nn.Parameters are
# identical across folds) plus a small trained head (Dropout -> Linear).
# Only two things can differ per fold: (1) the head's weights, since each
# fold trains a freshly-initialized head from scratch on a different
# train/val split, and (2) for architectures whose frozen backbone contains
# BatchNorm (RegNet-Y-16GF, EfficientNet-B3), the running_mean/running_var
# buffers -- frozen conv weights don't update during training, but BatchNorm
# buffers still drift via the forward pass alone whenever the module is in
# .train() mode, so different folds' data can leave slightly different
# buffer values even though the learnable weights never moved. Averaging
# the WHOLE state dict (not just the head) handles both correctly rather
# than silently ignoring the second effect.
# --------------------------------------------------------------------------
def uniform_soup_state_dict(model_key: str, folds: list = None, device: torch.device = DEVICE) -> dict:
    """Elementwise mean of the given folds' checkpoints, key by key.
    Integer buffers (e.g. BatchNorm's num_batches_tracked, a training-step
    counter) aren't meaningfully "averageable" -- kept from the first fold
    instead of blended."""
    folds = folds or list(range(1, N_FOLDS + 1))
    state_dicts = []
    for f in folds:
        path = _checkpoint_path(model_key, f)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint for {model_key} fold {f}: {path}")
        state_dicts.append(torch.load(path, map_location=device))

    keys = list(state_dicts[0].keys())
    for f, sd in zip(folds[1:], state_dicts[1:]):
        if list(sd.keys()) != keys:
            raise RuntimeError(f"{model_key} fold {f}'s state_dict keys differ from fold {folds[0]}'s -- can't soup")

    averaged = {}
    for k in keys:
        tensors = [sd[k] for sd in state_dicts]
        if tensors[0].is_floating_point():
            averaged[k] = torch.stack([t.float() for t in tensors], dim=0).mean(dim=0)
        else:
            averaged[k] = tensors[0]
    return averaged


def predict_soup(model_key: str, folds: list = None, device: torch.device = DEVICE, use_cache: bool = True) -> pd.DataFrame:
    """Same return contract as predict_checkpoint(), but for a weight-
    averaged "soup" of the given folds rather than one saved checkpoint."""
    folds = folds or list(range(1, N_FOLDS + 1))
    soup_tag = "soup" + "".join(str(f) for f in folds)
    cache_key = (model_key, soup_tag)
    if cache_key in _PRED_CACHE:
        return _PRED_CACHE[cache_key].copy()

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"model_key must be one of {list(MODEL_REGISTRY)}, got {model_key!r}")

    spec = MODEL_REGISTRY[model_key]
    cache_csv = PRED_CACHE_DIR / f"{spec['model_name']}_{soup_tag}_test_probs.csv"
    if use_cache and cache_csv.exists():
        df = pd.read_csv(cache_csv)
        _PRED_CACHE[cache_key] = df
        return df.copy()

    arch_config = ARCHITECTURE_REGISTRY[spec["arch_name"]]
    image_size = arch_config["input_size"]

    averaged_state = uniform_soup_state_dict(model_key, folds=folds, device=device)
    model = arch_config["build_fn"](0.0).to(device)
    model.load_state_dict(averaged_state)

    df = _run_inference(model, spec["tissue_type"], image_size, device, f"{model_key} soup{folds}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _PRED_CACHE[cache_key] = df
    if use_cache:
        PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df.copy()


def evaluate_soup(model_key: str, folds: list = None, use_cache: bool = True) -> dict:
    folds = folds or list(range(1, N_FOLDS + 1))
    df = predict_soup(model_key, folds=folds, use_cache=use_cache)
    label = f"{model_key}_soup_folds{''.join(str(f) for f in folds)}"
    return {"members": [(model_key, f"soup{folds}")], "label": label, "metrics": evaluate_predictions(df), "predictions": df}


# --------------------------------------------------------------------------
# Greedy soup (Wortsman et al. 2022): candidates tried in descending order
# of their own individual validation score, each tentatively merged into
# the running soup and kept only if the running soup's evaluation score
# doesn't drop, else discarded.
#
# READ BEFORE TRUSTING final_test_f1 -- this project's 3-fold split makes a
# leak-free multi-fold evaluation set impossible: every pool patient trains
# in 2 of the 3 folds and validates in only the third, so no patient is
# EVER held out from two or more folds simultaneously. The only
# contamination-free set available for a >1-fold accept/reject decision is
# the sealed TEST set itself -- so that's what this function uses, since
# there is no cleaner option with the data this project has. But that means
# the keep/discard decisions are made BY LOOKING AT the test set, which the
# final soup is then also scored on -- classic selection leakage. Every
# other number in this project (including uniform_soup_state_dict's soups)
# is an honest, unbiased test-set evaluation; this one is not. Report
# final_test_f1 as "the best subset this specific 33-patient test set
# happened to reward", not as this architecture's true expected
# performance -- a fair greedy soup would need a 4th, dedicated held-out
# split that this project's checkpoints were never trained to leave aside.
# --------------------------------------------------------------------------
def greedy_soup(model_key: str, folds: list = None, use_cache: bool = True, verbose: bool = True) -> dict:
    folds = folds or list(range(1, N_FOLDS + 1))
    spec = MODEL_REGISTRY[model_key]
    folds_csv = LOGS_DIR / f"{spec['model_name']}_kfold_folds.csv"
    folds_df = pd.read_csv(folds_csv).set_index("fold")

    # Ordering by each fold's OWN best_val_f1 is leak-free (it's exactly
    # the quantity already used to pick that fold's checkpoint during
    # training) -- only the accept/reject decision below touches test data.
    order = sorted(folds, key=lambda f: folds_df.loc[f, "best_val_f1"], reverse=True)

    trace = []
    current = [order[0]]
    current_f1 = evaluate_soup(model_key, folds=current, use_cache=use_cache)["metrics"]["overall"]["f1"]
    trace.append({"fold_tried": order[0], "own_val_f1": float(folds_df.loc[order[0], "best_val_f1"]), "action": "seed", "folds_after": list(current), "test_f1_after": current_f1})
    if verbose:
        print(f"  [greedy_soup:{model_key}] seed fold {order[0]} (own best_val_f1={folds_df.loc[order[0], 'best_val_f1']:.4f}) -> soup test_f1={current_f1:.4f}")

    for f in order[1:]:
        trial_folds = current + [f]
        trial_f1 = evaluate_soup(model_key, folds=trial_folds, use_cache=use_cache)["metrics"]["overall"]["f1"]
        keep = trial_f1 >= current_f1  # >= (not >) -- on a 33-patient test set, ties are common; favor the larger, more-averaged soup on a tie
        trace.append({"fold_tried": f, "own_val_f1": float(folds_df.loc[f, "best_val_f1"]), "action": "kept" if keep else "discarded", "folds_after": list(trial_folds if keep else current), "test_f1_after": trial_f1 if keep else current_f1})
        if verbose:
            print(f"  [greedy_soup:{model_key}] try fold {f} (own best_val_f1={folds_df.loc[f, 'best_val_f1']:.4f}) -> soup test_f1={trial_f1:.4f} vs current={current_f1:.4f} -- {'KEPT' if keep else 'discarded'}")
        if keep:
            current, current_f1 = trial_folds, trial_f1

    return {
        "model_key": model_key,
        "candidate_order_by_own_val_f1": order,
        "final_folds": current,
        "final_test_f1": current_f1,
        "trace": trace,
    }


def verify_soup_degenerates_to_checkpoint(model_key: str, fold: int, tol: float = 1e-4) -> None:
    """A soup over a SINGLE fold is just that fold's own weights (mean of
    one item = itself) -- so predict_soup([f]) must reproduce
    predict_checkpoint(f)'s probabilities almost exactly. The two code
    paths differ (fp16->fp32 load + no averaging vs. fp16->fp32 load +
    torch.stack().mean() over one element), so "almost" rather than bit-
    identical guards against a real bug in uniform_soup_state_dict without
    being defeated by that harmless float rounding difference."""
    single = predict_checkpoint(model_key, fold).set_index("patient_id").sort_index()
    souped = predict_soup(model_key, folds=[fold]).set_index("patient_id").sort_index()
    max_diff = (single["prob"] - souped["prob"]).abs().max()
    if max_diff > tol:
        raise RuntimeError(
            f"{model_key}: soup([{fold}]) should reproduce predict_checkpoint({fold}) almost exactly "
            f"(max prob diff={max_diff:.6f} > tol={tol}) -- uniform_soup_state_dict() likely has a bug."
        )
    print(f"  [verified] {model_key}: soup([{fold}]) degenerates to fold {fold}'s own predictions (max diff={max_diff:.2e}).")


# --------------------------------------------------------------------------
# Stacking (out-of-fold / OOF): train a small meta-model on base
# architectures' predictions, instead of a fixed average (soft voting) or a
# fixed weight-average (soup).
#
# The leak-free trick this project's K-fold structure enables for free:
# fold f's checkpoint never saw fold f's own val_ids during training -- so
# running fold f's checkpoint on ONLY fold f's own val_ids gives genuinely
# out-of-fold predictions. Since the 3 folds' val_ids partition the entire
# pool with no overlap (build_folds() is a StratifiedKFold split), doing
# this for all 3 folds and concatenating covers every pool patient exactly
# once, with zero leakage -- the standard "OOF" recipe used for stacking in
# every serious ML competition. This is a fundamentally different, cleaner
# situation than greedy_soup() faced: that needed a validation set for a
# MULTI-fold combination (which doesn't exist leak-free here); this only
# ever evaluates ONE fold's checkpoint on ITS OWN held-out patients, which
# is exactly what K-fold CV already guarantees is clean.
#
# The meta-learner (logistic regression) is fit on these OOF probabilities
# vs. true pool labels -- never touching the sealed test set. At test time,
# each base architecture's prediction is its fold-averaged (soft-voted)
# probability (evaluate_members(all_folds(key)), reused from above) -- the
# standard way to apply a K-fold-trained base learner to genuinely new
# data. So both the meta-learner's training (OOF, no test leakage) and its
# final evaluation (test set, no training leakage) are honest -- unlike
# greedy_soup, nothing here is selection-biased.
# --------------------------------------------------------------------------
def predict_oof(model_key: str, device: torch.device = DEVICE, use_cache: bool = True) -> pd.DataFrame:
    """Out-of-fold predictions covering the whole pool (every train+val
    patient for this architecture's tissue_type) -- one row per pool
    patient, using the ONE fold that held that patient out."""
    cache_key = (model_key, "oof")
    if cache_key in _PRED_CACHE:
        return _PRED_CACHE[cache_key].copy()

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"model_key must be one of {list(MODEL_REGISTRY)}, got {model_key!r}")

    spec = MODEL_REGISTRY[model_key]
    cache_csv = PRED_CACHE_DIR / f"{spec['model_name']}_oof_pool_probs.csv"
    if use_cache and cache_csv.exists():
        df = pd.read_csv(cache_csv)
        _PRED_CACHE[cache_key] = df
        return df.copy()

    pool_df = load_pool_df(spec["tissue_type"])
    folds = build_folds(pool_df)  # same seed=42, same pool_df row order as kfold_engine.run_kfold_study -- reproduces the exact fold assignment used to train these checkpoints
    arch_config = ARCHITECTURE_REGISTRY[spec["arch_name"]]
    image_size = arch_config["input_size"]

    fold_dfs = []
    for fold_idx, (_train_ids, val_ids) in enumerate(folds):
        fold_num = fold_idx + 1
        checkpoint_path = _checkpoint_path(model_key, fold_num)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint for {model_key} fold {fold_num}: {checkpoint_path}")

        model = arch_config["build_fn"](0.0).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        fold_df = _run_inference(model, spec["tissue_type"], image_size, device, f"{model_key} OOF fold {fold_num}", patient_ids=val_ids)
        fold_df["fold"] = fold_num
        fold_dfs.append(fold_df)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_df = pd.concat(fold_dfs, ignore_index=True)
    if not oof_df["patient_id"].is_unique:
        raise RuntimeError(
            f"{model_key}: OOF patient_id has duplicates -- fold val_ids overlapped, which should be "
            f"impossible for a StratifiedKFold partition. build_folds() reproduction likely diverged "
            f"from what actually trained these checkpoints."
        )

    _PRED_CACHE[cache_key] = oof_df
    if use_cache:
        PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(cache_csv, index=False)
    return oof_df.copy()


def verify_oof_fold_sizes(model_key: str) -> None:
    """Cheap, decisive correctness check: build_folds()'s reproduced
    val_ids sizes must match this architecture's recorded n_val per fold
    (kfold_folds.csv) exactly -- if they don't, the fold reproduction here
    has silently diverged from what actually trained the checkpoints (e.g.
    a pool_df ordering difference), and predict_oof()'s "no leakage"
    guarantee would be void."""
    spec = MODEL_REGISTRY[model_key]
    folds_csv = LOGS_DIR / f"{spec['model_name']}_kfold_folds.csv"
    recorded = pd.read_csv(folds_csv).set_index("fold")["n_val"].to_dict()

    pool_df = load_pool_df(spec["tissue_type"])
    folds = build_folds(pool_df)
    for fold_idx, (_train_ids, val_ids) in enumerate(folds):
        fold_num = fold_idx + 1
        if len(val_ids) != recorded[fold_num]:
            raise RuntimeError(
                f"{model_key} fold {fold_num}: reproduced val_ids has {len(val_ids)} patients, "
                f"recorded n_val={recorded[fold_num]} -- build_folds() reproduction has diverged, "
                f"do not trust predict_oof() until this matches exactly."
            )
    print(f"  [verified] {model_key}: reproduced fold sizes match recorded n_val exactly -- OOF split is trustworthy.")


def fit_stacking_meta_learner(model_keys: list, use_cache: bool = True):
    """Fits a LogisticRegression meta-learner on OOF probabilities from the
    given base architectures vs. true pool labels. Returns (meta_model,
    common_patient_ids). Architectures with a smaller pool (efficientnet_b3
    -- 6 patients lack a forniceal_palpebral crop) are handled by
    restricting to the intersection of every base learner's OOF patient_id
    set, not by erroring."""
    from sklearn.linear_model import LogisticRegression

    oof_dfs = {k: predict_oof(k, use_cache=use_cache).set_index("patient_id").sort_index() for k in model_keys}
    common_ids = sorted(set.intersection(*(set(df.index) for df in oof_dfs.values())))
    if not common_ids:
        raise RuntimeError(f"No pool patients are common to every base learner in {model_keys}")

    label_cols = [oof_dfs[k].loc[common_ids, "label"].to_numpy() for k in model_keys]
    if not all(np.array_equal(label_cols[0], lc) for lc in label_cols[1:]):
        raise RuntimeError(f"OOF label disagreement across base learners {model_keys} for some patient_id -- data bug")

    X = np.column_stack([oof_dfs[k].loc[common_ids, "prob"].to_numpy() for k in model_keys])
    y = label_cols[0]

    meta = LogisticRegression(max_iter=1000)
    meta.fit(X, y)
    return meta, common_ids


def evaluate_stacking(model_keys: list, use_cache: bool = True) -> dict:
    """Fits the meta-learner on OOF pool predictions (see above), then
    applies it to each base architecture's fold-averaged TEST-set
    prediction (evaluate_members(all_folds(key))) to get the final stacked
    test-set metrics. Both halves are leak-free."""
    meta, _common_pool_ids = fit_stacking_meta_learner(model_keys, use_cache=use_cache)

    test_dfs = {k: evaluate_members(all_folds(k), use_cache=use_cache)["predictions"].set_index("patient_id").sort_index() for k in model_keys}
    common_test_ids = sorted(set.intersection(*(set(df.index) for df in test_dfs.values())))

    label_cols = [test_dfs[k].loc[common_test_ids, "label"].to_numpy() for k in model_keys]
    if not all(np.array_equal(label_cols[0], lc) for lc in label_cols[1:]):
        raise RuntimeError(f"Test-set label disagreement across base learners {model_keys} for some patient_id -- data bug")

    X_test = np.column_stack([test_dfs[k].loc[common_test_ids, "prob"].to_numpy() for k in model_keys])
    y_test = label_cols[0]
    countries = test_dfs[model_keys[0]].loc[common_test_ids, "country"].to_numpy()

    stacked_probs = meta.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, stacked_probs, countries)

    return {
        "model_keys": model_keys,
        "meta_coef": dict(zip(model_keys, meta.coef_[0].tolist())),
        "meta_intercept": float(meta.intercept_[0]),
        "n_test_patients": len(common_test_ids),
        "metrics": metrics,
    }


def best_fold_number(model_key: str) -> int:
    """Picks the fold whose validation F1 (best_val_f1, the same quantity
    used for checkpoint selection during training) was highest -- i.e. model
    selection uses only information that would have been available without
    peeking at the sealed test set. Reads the already-saved
    {model_name}_kfold_folds.csv rather than re-deriving it."""
    spec = MODEL_REGISTRY[model_key]
    folds_csv = LOGS_DIR / f"{spec['model_name']}_kfold_folds.csv"
    folds_df = pd.read_csv(folds_csv)
    return int(folds_df.loc[folds_df["best_val_f1"].idxmax(), "fold"])


def best_fold(model_key: str) -> list:
    """[(model_key, best_fold_number)] -- a 1-member list, so it composes
    with + like all_folds() below."""
    return [(model_key, best_fold_number(model_key))]


def all_folds(model_key: str) -> list:
    """[(model_key, 1), (model_key, 2), (model_key, 3)]."""
    return [(model_key, f) for f in range(1, N_FOLDS + 1)]


# --------------------------------------------------------------------------
# Combining predictions + evaluating
# --------------------------------------------------------------------------
def combine_predictions(member_dfs: list, weights: list = None) -> pd.DataFrame:
    """Soft-voting merge: average each member's probability per patient_id
    (weighted if `weights` given, else equal weight per checkpoint -- note
    this means an architecture contributing 3 folds to a mixed ensemble
    counts 3x as heavily as one contributing only its best fold; that is a
    deliberate "one vote per checkpoint" choice, not "one vote per
    architecture", kept simple for this first version). Merges on
    patient_id explicitly (not positionally), and asserts every member
    agrees on label/country per patient -- both are ground truth, so any
    disagreement would mean two members disagree about the sealed test set
    itself, a data bug worth failing loudly on rather than averaging over."""
    if not member_dfs:
        raise ValueError("combine_predictions() needs at least one member")
    if weights is not None and len(weights) != len(member_dfs):
        raise ValueError(f"weights has {len(weights)} entries but there are {len(member_dfs)} members")

    base = member_dfs[0].set_index("patient_id").sort_index()
    prob_matrix = np.empty((len(base), len(member_dfs)))
    prob_matrix[:, 0] = base["prob"].to_numpy()

    for i, df in enumerate(member_dfs[1:], start=1):
        d = df.set_index("patient_id").sort_index()
        if not d.index.equals(base.index):
            raise ValueError(f"member {i} has a different patient_id set than member 0 -- can't merge by patient_id")
        if not (d["label"] == base["label"]).all() or not (d["country"] == base["country"]).all():
            raise ValueError(f"member {i} disagrees with member 0 on label/country for some patient_id")
        prob_matrix[:, i] = d["prob"].to_numpy()

    if weights is None:
        ensemble_prob = prob_matrix.mean(axis=1)
    else:
        w = np.asarray(weights, dtype=float)
        ensemble_prob = (prob_matrix * w).sum(axis=1) / w.sum()

    return pd.DataFrame(
        {
            "patient_id": base.index,
            "country": base["country"].to_numpy(),
            "label": base["label"].to_numpy(),
            "prob": ensemble_prob,
        }
    )


def evaluate_predictions(df: pd.DataFrame) -> dict:
    return compute_metrics(df["label"].to_numpy(), df["prob"].to_numpy(), df["country"].to_numpy())


def evaluate_members(members: list, weights: list = None, use_cache: bool = True) -> dict:
    """members: list of (model_key, fold) tuples, e.g.
    best_fold("maxvit_t") + best_fold("regnet_y_16gf"). Returns
    {"members": [...], "metrics": {...}, "predictions": DataFrame}."""
    member_dfs = [predict_checkpoint(k, f, use_cache=use_cache) for k, f in members]
    combined = combine_predictions(member_dfs, weights=weights)
    return {"members": list(members), "metrics": evaluate_predictions(combined), "predictions": combined}


# --------------------------------------------------------------------------
# Correctness check: recomputed single-checkpoint metrics vs. the recorded
# {model_name}_kfold_folds.csv row for that fold.
#
# f1/accuracy are threshold-derived (>0.5) so they reproduce bit-for-bit
# given the same predicted class per patient. auc is rank-based over raw
# probabilities, so it is sensitive to sub-percent probability jitter that
# never flips a threshold decision -- and some jitter is expected here:
# these checkpoints were originally trained/evaluated on a Kaggle GPU, and
# this script re-runs inference on local hardware (RTX 4050). Different
# GPUs can select different cuDNN convolution kernels for the same op,
# whose summation order differs at the floating-point level -- a real,
# known source of run-to-run numerical drift, not a determinism bug in this
# code. Confirmed empirically: coatnet_3 fold 1 reproduced test_f1/accuracy
# exactly but test_auc off by ~0.0038 (0.8421 recomputed vs 0.8383
# recorded) -- on a 33-patient test set, a probability shift too small to
# move any prediction across 0.5 can still flip one pairwise ranking and
# move AUC by roughly 1/n_pairs. f1/accuracy therefore use a tight
# tolerance (must match the training run's own recorded numbers exactly);
# auc uses a looser one, since the sealed test set is small enough that this
# specific kind of hardware drift is expected to show up there first.
# --------------------------------------------------------------------------
def verify_against_recorded(model_key: str, fold: int, tol: float = 1e-6, auc_tol: float = 0.01) -> None:
    spec = MODEL_REGISTRY[model_key]
    folds_csv = LOGS_DIR / f"{spec['model_name']}_kfold_folds.csv"
    recorded = pd.read_csv(folds_csv)
    row = recorded.loc[recorded["fold"] == fold].iloc[0]

    result = evaluate_members([(model_key, fold)])
    m = result["metrics"]["overall"]

    checks = [("test_f1", m["f1"], tol), ("test_accuracy", m["accuracy"], tol), ("test_auc", m["auc"], auc_tol)]
    for col, recomputed, this_tol in checks:
        recorded_val = row[col]
        if abs(recomputed - recorded_val) > this_tol:
            raise RuntimeError(
                f"{model_key} fold {fold}: recomputed {col}={recomputed:.6f} != "
                f"recorded {recorded_val:.6f} (kfold_folds.csv), diff={abs(recomputed - recorded_val):.6f} > "
                f"tol={this_tol}. Inference path likely wrong (tissue_type/image_size/checkpoint "
                f"mismatch) -- do not trust ensemble results until this reproduces within tolerance."
            )
    print(f"  [verified] {model_key} fold {fold}: recomputed test_f1/accuracy/auc match {folds_csv.name} (within tolerance).")


# --------------------------------------------------------------------------
# Full comparison run: baselines (every individual checkpoint) + named
# ensembles, saved as JSON/CSV/plot.
# --------------------------------------------------------------------------
def _flatten_metrics_row(name: str, kind: str, n_members: int, metrics: dict) -> dict:
    row = {"name": name, "kind": kind, "n_members": n_members}
    for bucket, prefix in [("overall", ""), ("India", "india_"), ("Italy", "italy_")]:
        b = metrics.get(bucket, {})
        row[f"{prefix}f1"] = b.get("f1")
        row[f"{prefix}accuracy"] = b.get("accuracy")
        row[f"{prefix}auc"] = b.get("auc")
        row[f"{prefix}sensitivity"] = b.get("sensitivity")
        row[f"{prefix}specificity"] = b.get("specificity")
    return row


def run_comparison(
    ensemble_defs: dict,
    soup_defs: dict = None,
    greedy_soup_keys: list = None,
    stacking_defs: dict = None,
    verify_baselines: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """ensemble_defs: {name: [(model_key, fold), ...]} (soft-voting, one
    forward pass per member). soup_defs: {name: (model_key, [folds])}
    (weight-averaging, one forward pass total per soup). greedy_soup_keys:
    [model_key, ...] to run greedy_soup() on -- kept in its own "kind" and
    its own comparison_df/JSON, since (see greedy_soup()'s docstring) its
    result is test-selected/optimistic, not an unbiased number like every
    other row here. stacking_defs: {name: [model_key, ...]} -- OOF stacking
    (see evaluate_stacking()), leak-free like every "kind" except
    greedy_soup. Evaluates every individual checkpoint referenced by any
    ensemble as a baseline row (deduplicated), then every named ensemble,
    then every named soup, then any greedy soups, then any named stackings,
    then saves Output/version2/logs/ensemble_comparison.{csv,json} and
    Output/version2/plots/ensemble_comparison.png. Returns the comparison
    DataFrame."""
    verify_shared_test_set()
    soup_defs = soup_defs or {}
    greedy_soup_keys = greedy_soup_keys or []
    stacking_defs = stacking_defs or {}

    baseline_members = sorted({member for members in ensemble_defs.values() for member in members})
    print(f"\n=== Baselines: {len(baseline_members)} individual checkpoints ===")
    rows = []
    for model_key, fold in baseline_members:
        if verify_baselines:
            verify_against_recorded(model_key, fold)
        result = evaluate_members([(model_key, fold)], use_cache=use_cache)
        rows.append(_flatten_metrics_row(f"{model_key}_fold{fold}", "baseline", 1, result["metrics"]))

    print(f"\n=== Ensembles: {len(ensemble_defs)} combinations ===")
    ensemble_results = {}
    for name, members in ensemble_defs.items():
        result = evaluate_members(members, use_cache=use_cache)
        ensemble_results[name] = result
        m = result["metrics"]["overall"]
        print(f"  {name:36s} n={len(members):2d}  test_f1={m['f1']:.4f}  test_acc={m['accuracy']:.4f}  test_auc={m['auc']:.4f}")
        rows.append(_flatten_metrics_row(name, "ensemble", len(members), result["metrics"]))

    soup_results = {}
    if soup_defs:
        print(f"\n=== Soups: {len(soup_defs)} weight-averaged single models ===")
        soup_folds_checked = set()
        for name, (model_key, folds) in soup_defs.items():
            if verify_baselines and (model_key, folds[0]) not in soup_folds_checked:
                verify_soup_degenerates_to_checkpoint(model_key, folds[0])
                soup_folds_checked.add((model_key, folds[0]))
            result = evaluate_soup(model_key, folds=folds, use_cache=use_cache)
            soup_results[name] = result
            m = result["metrics"]["overall"]
            print(f"  {name:36s} folds={folds}  test_f1={m['f1']:.4f}  test_acc={m['accuracy']:.4f}  test_auc={m['auc']:.4f}")
            rows.append(_flatten_metrics_row(name, "soup", len(folds), result["metrics"]))

    greedy_results = {}
    if greedy_soup_keys:
        print(f"\n=== Greedy soups: {len(greedy_soup_keys)} architectures (TEST-SELECTED -- see greedy_soup() docstring) ===")
        for model_key in greedy_soup_keys:
            gs = greedy_soup(model_key, use_cache=use_cache)
            greedy_results[model_key] = gs
            final_result = evaluate_soup(model_key, folds=gs["final_folds"], use_cache=use_cache)
            name = f"{model_key}_greedy_soup_TEST_SELECTED"
            rows.append(_flatten_metrics_row(name, "greedy_soup", len(gs["final_folds"]), final_result["metrics"]))
            print(f"  {model_key}: final_folds={gs['final_folds']}  final_test_f1={gs['final_test_f1']:.4f}")

    stacking_results = {}
    if stacking_defs:
        print(f"\n=== Stacking: {len(stacking_defs)} meta-learner combinations (OOF, leak-free) ===")
        oof_checked = set()
        for name, model_keys in stacking_defs.items():
            if verify_baselines:
                for k in model_keys:
                    if k not in oof_checked:
                        verify_oof_fold_sizes(k)
                        oof_checked.add(k)
            result = evaluate_stacking(model_keys, use_cache=use_cache)
            stacking_results[name] = result
            m = result["metrics"]["overall"]
            coef_str = ", ".join(f"{k}={v:+.2f}" for k, v in result["meta_coef"].items())
            print(f"  {name:36s} bases={model_keys}  test_f1={m['f1']:.4f}  test_acc={m['accuracy']:.4f}  test_auc={m['auc']:.4f}  coefs=[{coef_str}]")
            rows.append(_flatten_metrics_row(name, "stacking", len(model_keys), result["metrics"]))

    comparison_df = pd.DataFrame(rows)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOGS_DIR / "ensemble_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    json_path = LOGS_DIR / "ensemble_comparison.json"
    json_payload = {
        "ensembles": {
            name: {"members": [list(m) for m in r["members"]], "metrics": r["metrics"]}
            for name, r in ensemble_results.items()
        },
        "soups": {
            name: {"model_key": r["members"][0][0], "folds": soup_defs[name][1], "metrics": r["metrics"]}
            for name, r in soup_results.items()
        },
        "greedy_soups_TEST_SELECTED_optimistic": greedy_results,
        "stacking": stacking_results,
        "baselines": [row for row in rows if row["kind"] == "baseline"],
    }
    import json as _json

    with open(json_path, "w") as f:
        _json.dump(json_payload, f, indent=2)

    print(f"\nSaved comparison table to {csv_path}")
    print(f"Saved full results to {json_path}")

    plot_path = _plot_comparison(comparison_df)
    print(f"Saved comparison plot to {plot_path}")

    return comparison_df


def _plot_comparison(comparison_df: pd.DataFrame) -> Path:
    df = comparison_df.sort_values(["kind", "f1"], ascending=[True, False]).reset_index(drop=True)
    kind_colors = {
        "baseline": "tab:blue",
        "ensemble": "tab:orange",
        "soup": "tab:green",
        "greedy_soup": "tab:red",
        "stacking": "tab:purple",
    }
    colors = [kind_colors[k] for k in df["kind"]]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(df)), 6))
    x = np.arange(len(df))
    ax.bar(x, df["f1"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=60, ha="right")
    ax.set_ylabel("Test F1 (overall)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Individual checkpoints (blue) / soft-voting (orange) / uniform soup (green) / "
        "greedy soup, TEST-SELECTED-optimistic (red) -- sealed test set"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "ensemble_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
