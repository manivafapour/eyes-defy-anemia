"""
Phase 4 (Classification): PyTorch dataset + transforms.

Independent of scripts/dataset.py (the segmentation-phase module) -- this
reads exclusively from classification/data/processed/, produced by
classification/scripts/prepare_dataset.py.
"""

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODULE_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = MODULE_ROOT / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"
EXTRACTION_LOG_CSV = PROCESSED_DIR / "extraction_log.csv"

TISSUE_TYPES = ["palpebral", "forniceal_palpebral"]
IMAGE_SIZE = 256
BATCH_SIZE = 16
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, fill=0, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class TissueClassificationDataset(Dataset):
    """Returns (image, label, country). image is a palpebral or
    forniceal_palpebral crop (already flattened to a black background and
    padded/resized to 256x256 by prepare_dataset.py); label is the
    WHO-threshold anemic_label; country is returned alongside the label so
    callers can compute per-country stratified metrics (India vs Italy)
    without a second file lookup -- required by this phase's evaluation
    constraint (report metrics split by country to monitor the confound).

    forniceal_palpebral is missing for 6 patients (the source dataset's own
    documentation states the forniceal conjunctiva was not exposed in those
    photos) -- this class filters to patients with an "ok" extraction
    status for the requested tissue_type, joined from extraction_log.csv,
    without modifying splits.csv itself (that file is shared across both
    tissue-type variants, same pattern as the segmentation phase's aligned
    vs. crop-based datasets)."""

    def __init__(
        self,
        split: str,
        tissue_type: str,
        splits_csv: Path = SPLITS_CSV,
        images_dir: Path = IMAGES_DIR,
        extraction_log_csv: Path = EXTRACTION_LOG_CSV,
        transform=None,
    ):
        if tissue_type not in TISSUE_TYPES:
            raise ValueError(f"tissue_type must be one of {TISSUE_TYPES}, got {tissue_type!r}")

        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split]

        log = pd.read_csv(extraction_log_csv)
        ok_ids = set(log.loc[log[f"{tissue_type}_status"] == "ok", "patient_id"])
        df = df[df["patient_id"].isin(ok_ids)]

        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir) / tissue_type
        self.tissue_type = tissue_type
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

        return image, label, row["country"]


# --------------------------------------------------------------------------
# DataLoaders
# --------------------------------------------------------------------------
def get_dataloaders(tissue_type: str, batch_size: int = BATCH_SIZE, num_workers: int = 0) -> dict:
    train_tf = get_train_transforms()
    eval_tf = get_eval_transforms()

    datasets = {
        "train": TissueClassificationDataset(split="train", tissue_type=tissue_type, transform=train_tf),
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
        print(f"\n--- {tissue} train batch ---")
        print("image shape [B, C, H, W]:", tuple(images.shape))
        print("label shape [B]:", tuple(labels.shape))
        print("label values:", labels.tolist())
        print("countries:", list(countries))
        for split, ds in [("train", loaders["train"].dataset), ("val", loaders["val"].dataset), ("test", loaders["test"].dataset)]:
            print(f"  {split}: {len(ds)} patients")
