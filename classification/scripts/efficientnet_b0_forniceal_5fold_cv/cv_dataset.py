"""
EfficientNet-B0 (forniceal_palpebral) -- 5-Fold Cross-Validation pipeline.
Dataloader module.

Standalone for this dedicated experiment, but reuses the shared transform
builders and resolution constant from ../dataset.py (classification's
shared v2 pipeline) rather than duplicating them -- the augmentation
baseline is unchanged by explicit decision (HorizontalFlip + Rotate only,
no color/brightness jitter, since that risks distorting the diagnostic
pallor signal itself; see classification/.project_memory/04_literature_review_findings.md).

Named cv_dataset.py, not dataset.py: this file and ../dataset.py both need
to be importable in the same process (cv_trainer_engine.py imports
functions from the shared ../trainer_engine.py, which itself imports from
../dataset.py under the bare module name "dataset"). Two files both named
dataset.py on sys.path at once would collide in Python's module cache --
whichever loaded first would silently satisfy every later `import dataset`,
including from the other one's shared dependency. Different filenames
sidestep this entirely.

Test-set isolation: the 33 patients with split=="test" in splits.csv are
never loaded by anything in this module -- excluded once in load_cv_pool()
before fold construction even starts, not filtered out per-fold. There is
no code path here that can see them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

SHARED_SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # classification/scripts/
sys.path.insert(0, str(SHARED_SCRIPTS_DIR))
from dataset import get_train_transforms, get_eval_transforms, IMAGE_SIZE  # noqa: E402 -- reuse, don't duplicate

MODULE_ROOT = SHARED_SCRIPTS_DIR.parent  # classification/
PROCESSED_DIR = MODULE_ROOT / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images" / "forniceal_palpebral"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"
EXTRACTION_LOG_CSV = PROCESSED_DIR / "extraction_log.csv"

TISSUE_TYPE = "forniceal_palpebral"
N_FOLDS = 5
SEED = 42


class PatientListDataset(Dataset):
    """Same (image, label, country) contract as the shared
    TissueClassificationDataset, but membership is an explicit patient
    DataFrame rather than a fixed split column -- required for k-fold CV,
    where each fold's train/val patient sets are different every fold and
    don't correspond to any single column in splits.csv."""

    def __init__(self, df: pd.DataFrame, images_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        image = np.array(Image.open(self.images_dir / f"{row['patient_id']}.jpg").convert("RGB"))
        label = torch.tensor(float(row["anemic_label"]), dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label, row["country"]


def load_cv_pool() -> pd.DataFrame:
    """Returns the forniceal_palpebral-eligible train+val pool (178
    patients), with the 33 locked-away test patients excluded before
    anything else touches the data.

    Not 184: 184 is the raw train+val patient count (151+33) before
    filtering to patients who actually have a forniceal_palpebral crop --
    6 of 217 patients dataset-wide lack one (documented: the source
    dataset's own notes say the forniceal conjunctiva wasn't exposed in
    those photos). Of those 6, 4 fall in the original train split and 2 in
    val, leaving 184-6=178 real, usable patients. Verified by
    cross-tabulating splits.csv against extraction_log.csv directly, not
    assumed -- smallest resulting stratum (Italy, anemic) is 19 patients,
    comfortably above the 5 required for 5-fold stratification.
    """
    splits = pd.read_csv(SPLITS_CSV)
    log = pd.read_csv(EXTRACTION_LOG_CSV)

    ok_ids = set(log.loc[log[f"{TISSUE_TYPE}_status"] == "ok", "patient_id"])
    pool = splits[splits["split"].isin(["train", "val"])].copy()  # test excluded here, permanently
    pool = pool[pool["patient_id"].isin(ok_ids)].reset_index(drop=True)
    return pool


def build_folds(pool: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> list:
    """Patient-level StratifiedKFold on the country+label compound key --
    the same stratification discipline as the project's original
    train/val/test split (prepare_dataset.py's create_patient_splits),
    applied here across folds instead of a single train/val cut. Returns a
    list of n_folds (train_df, val_df) tuples."""
    strata = pool["country"] + "_" + pool["anemic_label"].astype(int).astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in skf.split(pool, strata):
        train_df = pool.iloc[train_idx].reset_index(drop=True)
        val_df = pool.iloc[val_idx].reset_index(drop=True)
        folds.append((train_df, val_df))
    return folds


def get_fold_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, batch_size: int, num_workers: int = 0) -> dict:
    # Baseline augmentation, unchanged by explicit decision: HorizontalFlip
    # + Rotate(+-15 deg) only, no color/brightness jitter.
    train_tf = get_train_transforms(IMAGE_SIZE)
    eval_tf = get_eval_transforms(IMAGE_SIZE)

    train_ds = PatientListDataset(train_df, IMAGES_DIR, transform=train_tf)
    val_ds = PatientListDataset(val_df, IMAGES_DIR, transform=eval_tf)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


if __name__ == "__main__":
    pool = load_cv_pool()
    print(f"CV pool: {len(pool)} patients (test set of 33 excluded)")
    print(pool.groupby(["country", "anemic_label"]).size())

    folds = build_folds(pool)
    for i, (train_df, val_df) in enumerate(folds, start=1):
        print(f"\nFold {i}: train={len(train_df)} val={len(val_df)}")
        print("  train strata:", dict(train_df.groupby(["country", "anemic_label"]).size()))
        print("  val strata:  ", dict(val_df.groupby(["country", "anemic_label"]).size()))
