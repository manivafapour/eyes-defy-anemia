"""
new_way/: dataloader for the offline, country-stratified label-balanced
TRAIN+VAL pool (Offline_data_augmentation/generate_balanced_dataset.py's
output), built 2026-08-12 per the project author's explicit request to
attack the country<->label shortcut this project's memory repeatedly flags
-- see Offline_data_augmentation/generate_balanced_dataset.py's module
docstring for the full rationale.

Widened 2026-08-12 (same day as new_way/kfold/'s 3-fold retraining effort)
from a TRAIN-only manifest to a TRAIN+VAL pool manifest, to support K-fold
CV over the pooled patients. This changes what BalancedTissueClassification
Dataset's default (no patient_ids given) behavior means: previously every
manifest row was train-split-derived, so filtering to "just the train
split" was implicitly a no-op; now that val-origin patients are in the
manifest too, the default explicitly re-filters to the original "train"
split's patients (via splits.csv) to keep this class's own default,
single-split behavior byte-identical to before the widening. K-fold usage
always passes an explicit patient_ids list (a fold's own train patient
set, which may include former "val" patients) and bypasses that default
entirely -- see the class docstring below.

VAL and TEST are never touched by the rebalancing and are loaded here
completely unmodified, reusing datapreparepipeline.dataset.
TissueClassificationDataset directly (no duplicated loading logic) --
rebalancing eval data would invalidate every metric computed against it.
Transforms (get_train_transforms/get_eval_transforms, ImageNet
normalization) are imported from the same shared module too, for the same
reason: this project's rule is reuse existing pipeline code, don't fork it.

Online augmentation (HorizontalFlip/Rotate in get_train_transforms) is
still applied on top of the offline-balanced train images, deliberately --
it adds a further, different random transform on every epoch even to an
already-augmented offline copy, which is standard practice for combining
offline oversampling with online augmentation and reduces the risk of the
model memorizing the exact pixels of any one synthetic copy.

Named balanced_dataset.py, not dataset.py -- a bare "dataset" module name
would collide with datapreparepipeline/dataset.py in sys.modules whenever
both are imported in the same process (e.g. a training script that needs
both this loader and the shared trainer_engine.py, which itself imports
dataset internally): whichever loads first wins the cache, silently
serving the wrong one to every later `import dataset`. Verified this
collision actually occurs with the bare name before renaming, and that it
stops occurring with this name. Same fix this project already applied once
before, for the same reason (efficientnet_b0_forniceal_5fold_cv/'s
cv_dataset.py / cv_trainer_engine.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

NEW_WAY_DIR = Path(__file__).resolve().parent
DATAPREPAREPIPELINE_DIR = NEW_WAY_DIR.parent / "datapreparepipeline"
sys.path.insert(0, str(DATAPREPAREPIPELINE_DIR))

from dataset import (  # noqa: E402
    BATCH_SIZE,
    IMAGE_SIZE,
    SPLITS_CSV,
    TISSUE_TYPES,
    TissueClassificationDataset,
    get_eval_transforms,
    get_train_transforms,
)

OFFLINE_DIR = NEW_WAY_DIR / "Offline_data_augmentation"
OFFLINE_IMAGES_DIR = OFFLINE_DIR / "images"
MANIFEST_CSV = OFFLINE_DIR / "manifest.csv"


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class BalancedTissueClassificationDataset(Dataset):
    """Returns (image, label, country), the same 3-tuple contract as
    TissueClassificationDataset, and exposes self.df with an "anemic_label"
    column so it's a drop-in for trainer_engine.py's pos_weight computation
    (train_loader.dataset.df["anemic_label"]) -- on this balanced data
    pos_weight will come out close to 1.0 automatically, no special-casing
    needed.

    patient_ids (optional): an explicit list of patient IDs (matched against
    the manifest's source_patient_id, so both a patient's original row and
    any of its augmented copies are included/excluded together) -- added
    for new_way/kfold/'s K-fold engine, where a fold's training partition is
    a StratifiedKFold assignment over the pooled train+val patients, not a
    fixed split. When given, it completely overrides the default filtering
    below. When omitted (default), this class filters the manifest to only
    the patients whose real split (per splits.csv) is "train" -- this
    reproduces the original (unaugmented) row counts exactly (verified:
    India 19/47, Italy 69/16 for palpebral, byte-identical to before the
    widening), but NOT the exact augmented-image count, since balancing was
    computed once over the wider train+val pool and the augmented copies
    are distributed round-robin across ALL minority patients (train- and
    val-origin mixed) before this filter ever applies -- verified: palpebral
    default view now has 79 augmented images (India 28, Italy 51), not the
    81 (28/53) the original narrower generation produced. This is a small,
    expected, and understood deviation, not a bug -- pos_weight is computed
    dynamically per use anyway, so it self-corrects. Mirrors the identical
    patient_ids precedent already used in Segmentation/scripts/dataset.py's
    AlignedConjunctivaSegmentationDataset."""

    def __init__(
        self,
        tissue_type: str,
        manifest_csv: Path = MANIFEST_CSV,
        images_dir: Path = OFFLINE_IMAGES_DIR,
        transform=None,
        patient_ids: list = None,
    ):
        if tissue_type not in TISSUE_TYPES:
            raise ValueError(f"tissue_type must be one of {TISSUE_TYPES}, got {tissue_type!r}")
        if not Path(manifest_csv).exists():
            raise FileNotFoundError(
                f"{manifest_csv} not found -- run "
                f"Offline_data_augmentation/generate_balanced_dataset.py first."
            )

        manifest = pd.read_csv(manifest_csv)
        df = manifest[manifest["tissue_type"] == tissue_type]

        if patient_ids is not None:
            df = df[df["source_patient_id"].isin(set(patient_ids))]
        else:
            splits = pd.read_csv(SPLITS_CSV)
            train_ids = set(splits.loc[splits["split"] == "train", "patient_id"])
            df = df[df["source_patient_id"].isin(train_ids)]

        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"No rows for tissue_type={tissue_type!r} in {manifest_csv} after filtering.")

        self.df = df
        self.images_dir = Path(images_dir) / tissue_type
        self.tissue_type = tissue_type
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        image = np.array(Image.open(self.images_dir / f"{row['image_id']}.jpg").convert("RGB"))
        label = torch.tensor(float(row["anemic_label"]), dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label, row["country"]


# --------------------------------------------------------------------------
# DataLoaders
# --------------------------------------------------------------------------
def get_dataloaders(
    tissue_type: str, batch_size: int = BATCH_SIZE, num_workers: int = 0, image_size: int = IMAGE_SIZE
) -> dict:
    """Same signature/contract as datapreparepipeline.dataset.get_dataloaders
    (drop-in for trainer_engine.py's get_dataloaders_fn parameter) -- only
    "train" is sourced differently (the balanced offline set); "val"/"test"
    are the real, unmodified splits."""
    train_tf = get_train_transforms(image_size)
    eval_tf = get_eval_transforms(image_size)

    datasets = {
        "train": BalancedTissueClassificationDataset(tissue_type=tissue_type, transform=train_tf),
        "val": TissueClassificationDataset(split="val", tissue_type=tissue_type, transform=eval_tf),
        "test": TissueClassificationDataset(split="test", tissue_type=tissue_type, transform=eval_tf),
    }

    return {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers)
        for name, ds in datasets.items()
    }


# --------------------------------------------------------------------------
# Test block
# --------------------------------------------------------------------------
if __name__ == "__main__":
    for tissue in TISSUE_TYPES:
        loaders = get_dataloaders(tissue)
        images, labels, countries = next(iter(loaders["train"]))
        print(f"\n--- {tissue} balanced train batch ---")
        print("image shape [B, C, H, W]:", tuple(images.shape))
        print("label shape [B]:", tuple(labels.shape))
        print("label values:", labels.tolist())
        print("countries:", list(countries))
        for split, ds in [("train", loaders["train"].dataset), ("val", loaders["val"].dataset), ("test", loaders["test"].dataset)]:
            print(f"  {split}: {len(ds)} images")
        print("  train country x label breakdown:")
        print(loaders["train"].dataset.df.groupby(["country", "anemic_label"]).size().to_string().replace("\n", "\n    "))
