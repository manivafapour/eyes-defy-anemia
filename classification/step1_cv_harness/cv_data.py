"""
Step 1 -- Measurement Harness: pool construction, fold building, dataset.

Three responsibilities, kept separate from the training loop so the
structural verification in validate_harness.py can exercise all of them
WITHOUT training anything (checkpoints 1-4 and 6 are pure fold-geometry
checks and must be runnable in seconds, not hours).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from cv_config import (
    EXTRACTION_LOG_CSV,
    HELD_OUT_SPLIT,
    IMAGES_DIR,
    INNER_VAL_FRACTION,
    N_REPEATS,
    N_SPLITS,
    PIPELINE_DIR,
    POOL_SPLITS,
    SEED,
    SPLITS_CSV,
    STRATUM_COL,
)

sys.path.insert(0, str(PIPELINE_DIR))
# Transforms are imported from the live pipeline, not redefined -- the
# augmentation policy (HorizontalFlip p=0.5, Rotate +/-15, ImageNet Normalize)
# is part of what "the v2_clean model" means, and a local copy would drift.
from dataset import get_eval_transforms, get_train_transforms  # noqa: E402

TISSUE_TYPES = ["palpebral", "forniceal_palpebral"]


# --------------------------------------------------------------------------
# Pool construction
# --------------------------------------------------------------------------
def _add_stratum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[STRATUM_COL] = df["country"] + "_" + df["anemic_label"].astype(int).astype(str)
    return df


def _filter_to_available_crops(df: pd.DataFrame, tissue_type: str) -> pd.DataFrame:
    """forniceal_palpebral is missing for 6 patients dataset-wide (the source
    photos did not expose the forniceal conjunctiva). Which cells those 6 fall
    into determines that tissue type's real pair counts, so this is computed
    and reported per tissue type rather than assumed equal to palpebral's."""
    log = pd.read_csv(EXTRACTION_LOG_CSV)
    ok_ids = set(log.loc[log[f"{tissue_type}_status"] == "ok", "patient_id"])
    return df[df["patient_id"].isin(ok_ids)]


def load_pool(tissue_type: str, splits_csv: Path = SPLITS_CSV) -> pd.DataFrame:
    """The CV pool: train + val only. The test split is removed here, once,
    before any fold exists -- so no code path downstream of this function can
    reach a test patient even by accident."""
    if tissue_type not in TISSUE_TYPES:
        raise ValueError(f"tissue_type must be one of {TISSUE_TYPES}, got {tissue_type!r}")
    df = pd.read_csv(splits_csv)
    df = df[df["split"].isin(POOL_SPLITS)]
    df = _filter_to_available_crops(df, tissue_type)
    return _add_stratum(df).reset_index(drop=True)


def load_held_out(tissue_type: str, splits_csv: Path = SPLITS_CSV) -> pd.DataFrame:
    """The sealed test set. Loaded ONLY so validate_harness.py can assert its
    patient ids never appear in a fold. Nothing in Steps 1-5 may train or
    evaluate on it."""
    df = pd.read_csv(splits_csv)
    df = df[df["split"] == HELD_OUT_SPLIT]
    df = _filter_to_available_crops(df, tissue_type)
    return _add_stratum(df).reset_index(drop=True)


def pool_composition(df: pd.DataFrame) -> dict:
    """Per-cell counts plus the discordant-pair counts that set the precision
    ceiling for each country's AUC. The pair counts are the whole reason this
    harness exists, so they are computed and reported explicitly rather than
    left implicit."""
    counts = df[STRATUM_COL].value_counts().to_dict()
    out = {"n_total": int(len(df)), "cells": {k: int(v) for k, v in sorted(counts.items())}}
    for country in ("India", "Italy"):
        n_pos = int(counts.get(f"{country}_1", 0))
        n_neg = int(counts.get(f"{country}_0", 0))
        out[country] = {"n_anemic": n_pos, "n_healthy": n_neg, "n_discordant_pairs": n_pos * n_neg}
    n_pos_all = int(df["anemic_label"].sum())
    out["overall"] = {
        "n_anemic": n_pos_all,
        "n_healthy": int(len(df) - n_pos_all),
        "n_discordant_pairs": n_pos_all * (len(df) - n_pos_all),
    }
    return out


def shuffle_labels(df: pd.DataFrame, mode: str, seed: int = SEED) -> pd.DataFrame:
    """Label-shuffle negative control (checkpoint 5).

    'within_country' (default, and the more informative of the two): permutes
    labels inside each country separately. This PRESERVES each country's label
    marginal (India stays ~71.6% anemic, Italy ~18.9%) while destroying every
    per-patient association. Two things must then follow, and both are
    asserted: per-country AUC must collapse to 0.50, which proves the harness
    has no leakage; while overall AUC can still sit meaningfully above 0.50,
    which is a direct empirical demonstration of the country-shortcut
    mechanism -- a model that has learned nothing about any individual patient
    still scores well in aggregate purely because 62% of the overall AUC's
    pairs are cross-country.

    'global': permutes labels across the whole pool, destroying the
    country-label association too. Every AUC (overall and per-country) must
    then collapse to 0.50. Stricter as a pure leakage test, but it cannot
    demonstrate the shortcut mechanism.

    Strata are rebuilt from the SHUFFLED labels, so fold geometry stays
    identical in shape to a real run.
    """
    if mode not in ("within_country", "global"):
        raise ValueError(f"mode must be 'within_country' or 'global', got {mode!r}")
    rng = np.random.default_rng(seed)
    out = df.copy()
    if mode == "global":
        out["anemic_label"] = rng.permutation(out["anemic_label"].to_numpy())
    else:
        for country in out["country"].unique():
            mask = out["country"] == country
            out.loc[mask, "anemic_label"] = rng.permutation(out.loc[mask, "anemic_label"].to_numpy())
    return _add_stratum(out.drop(columns=[STRATUM_COL]))


# --------------------------------------------------------------------------
# Fold building
# --------------------------------------------------------------------------
def build_folds(
    pool_df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
) -> list:
    """Repeated stratified k-fold on the 4-cell country x label key.

    Returns a flat list of fold records, each with the outer train/val patient
    ids and the inner early-stopping split carved out of the training portion.
    Fold assignments are data, not a side effect: they are returned, persisted
    to disk by the runner, and reused verbatim by Steps 3-5 so that every
    later intervention can be compared to this baseline with a PAIRED
    bootstrap (far more powerful than checking whether two independent CIs
    overlap, which on 23 India-healthy patients would be hopeless).
    """
    y = pool_df[STRATUM_COL].to_numpy()
    patient_ids = pool_df["patient_id"].to_numpy()

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = []
    for global_idx, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(y)), y)):
        repeat = global_idx // n_splits
        fold = global_idx % n_splits

        # Inner split for early stopping only -- carved from the outer TRAIN
        # portion, never from the outer val fold (cv_config deviation 1).
        inner_train_idx, inner_val_idx = train_test_split(
            train_idx,
            test_size=INNER_VAL_FRACTION,
            random_state=seed + global_idx,
            stratify=y[train_idx],
        )
        folds.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "global_index": int(global_idx),
                "outer_train_ids": patient_ids[train_idx].tolist(),
                "outer_val_ids": patient_ids[val_idx].tolist(),
                "inner_train_ids": patient_ids[inner_train_idx].tolist(),
                "inner_val_ids": patient_ids[inner_val_idx].tolist(),
            }
        )
    return folds


# --------------------------------------------------------------------------
# Dataset over an explicit patient-id list
# --------------------------------------------------------------------------
class PatientListDataset(Dataset):
    """Returns (image, label, country, patient_id).

    Deliberately keyed on an explicit list of patient ids rather than on a
    split name (the live TissueClassificationDataset's contract) -- CV folds
    have no split name, and patient_id must come back with every prediction so
    out-of-fold predictions can be joined per patient for pooling and for the
    paired bootstrap in later steps.
    """

    def __init__(self, df: pd.DataFrame, patient_ids: list, tissue_type: str, transform=None):
        subset = df[df["patient_id"].isin(set(patient_ids))]
        self.df = subset.reset_index(drop=True)
        self.images_dir = Path(IMAGES_DIR) / tissue_type
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        image = np.array(Image.open(self.images_dir / f"{row['patient_id']}.jpg").convert("RGB"))
        label = torch.tensor(float(row["anemic_label"]), dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, label, row["country"], row["patient_id"]


def make_fold_loaders(
    pool_df: pd.DataFrame,
    fold: dict,
    tissue_type: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> dict:
    """Train loader gets the augmenting transform; both eval loaders get the
    deterministic one -- same asymmetry as the live pipeline."""
    train_tf = get_train_transforms(image_size)
    eval_tf = get_eval_transforms(image_size)

    specs = {
        "inner_train": (fold["inner_train_ids"], train_tf, True),
        "inner_val": (fold["inner_val_ids"], eval_tf, False),
        "outer_val": (fold["outer_val_ids"], eval_tf, False),
    }
    loaders = {}
    for name, (ids, tf, shuffle) in specs.items():
        ds = PatientListDataset(pool_df, ids, tissue_type, transform=tf)
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False
        )
    return loaders
