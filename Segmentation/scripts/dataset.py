"""
Phase 1: Dual Data Pipeline Construction for the Eyes-Defy-Anemia project.

Builds a patient-level, country+label-stratified train/val/test split on top
of the Phase 0 metadata, then exposes two PyTorch datasets:

- AlignedConjunctivaSegmentationDataset: (image, mask) pairs from
  data/processed/aligned_raw{,_forniceal}/{images,masks}/, built by
  scripts/build_aligned_dataset{,_forniceal}.py via SIFT/ORB + RANSAC
  homography alignment. image is the FULL raw photo; mask is a genuinely
  pixel-aligned tissue mask in that same coordinate frame -- the dataset for
  a segmentation model intended to generalize to raw photos (CLAUDE.md
  Sec 1.4). Supports both tissue types via `tissue_type`.
- AnemiaClassificationDataset: (image, label) pairs from
  data/processed/images/{patient_id}.jpg and the metadata's anemic_label.

(A third class, ConjunctivaSegmentationDataset -- crop-based, pairing each
palpebral crop with its own alpha channel -- existed here previously. It was
removed along with the three hand-built segmentation models that used it
(unet.py/attention_unet.py/resunet.py) once the pretrained-architecture
sweep superseded them; see CLAUDE.md Sec 1.2/2.1-2.5 for the historical
methodology record, which was intentionally left in place even though the
code is gone.)
"""

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"
SPLITS_CSV = PROCESSED_DIR / "dataset_splits.csv"
IMAGES_DIR = PROCESSED_DIR / "images"

ALIGNED_ROOT = PROCESSED_DIR / "aligned_raw"
ALIGNED_IMAGES_DIR = ALIGNED_ROOT / "images"
ALIGNED_MASKS_DIR = ALIGNED_ROOT / "masks"
ALIGNMENT_LOG_CSV = ALIGNED_ROOT / "alignment_log.csv"

ALIGNED_ROOT_FORNICEAL = PROCESSED_DIR / "aligned_raw_forniceal"

# Per-tissue-type resolution for AlignedConjunctivaSegmentationDataset's
# images_dir/masks_dir/alignment_log_csv defaults. "palpebral" reproduces
# this class's original (pre-tissue_type) behavior exactly, byte-for-byte
# path-for-path -- every existing caller that doesn't pass tissue_type is
# unaffected. "forniceal_palpebral" points at the sibling dataset built by
# scripts/build_aligned_dataset_forniceal.py (new -- see CLAUDE.md Sec 1.4
# follow-up on the 9-architecture x 2-tissue-type expansion).
ALIGNED_TISSUE_CONFIG = {
    "palpebral": {
        "images_dir": ALIGNED_IMAGES_DIR,
        "masks_dir": ALIGNED_MASKS_DIR,
        "alignment_log_csv": ALIGNMENT_LOG_CSV,
    },
    "forniceal_palpebral": {
        "images_dir": ALIGNED_ROOT_FORNICEAL / "images",
        "masks_dir": ALIGNED_ROOT_FORNICEAL / "masks",
        "alignment_log_csv": ALIGNED_ROOT_FORNICEAL / "alignment_log.csv",
    },
}

IMAGE_SIZE = 256
BATCH_SIZE = 16
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Stratified patient-level splitting
# --------------------------------------------------------------------------
def create_patient_splits(
    metadata_csv: Path = METADATA_CSV,
    output_csv: Path = SPLITS_CSV,
    seed: int = SEED,
) -> pd.DataFrame:
    """Patient-level 70/15/15 train/val/test split, stratified on
    country + anemic_label so both country class-balance and the known
    India/Italy demographic skew are preserved in every split."""
    df = pd.read_csv(metadata_csv)
    strata = df["country"] + "_" + df["anemic_label"].astype(int).astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=strata, random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=strata.loc[temp_df.index],
        random_state=seed,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    result = pd.concat([train_df, val_df, test_df]).sort_index()
    result.to_csv(output_csv, index=False)

    print(f"Wrote {len(result)} patient splits to {output_csv}")
    print(result.groupby(["split", "country"])["anemic_label"].agg(["count", "mean"]))
    return result


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Random horizontal flip + rotation, applied identically to image and
    mask (nearest-neighbor for the mask so it stays binary), then
    normalization/tensor conversion (mask is left untouched by Normalize)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=15,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Deterministic-only transforms for validation/test: no augmentation."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class AlignedConjunctivaSegmentationDataset(Dataset):
    """Returns (image, mask) from the raw-photo-aligned dataset
    (data/processed/aligned_raw/, built by scripts/build_aligned_dataset.py
    via SIFT/ORB + RANSAC homography -- CLAUDE.md Sec 1.4.2). image is the
    FULL raw clinical photo (data/processed/aligned_raw/images/{patient_id}.jpg)
    and mask is a genuinely pixel-aligned tissue mask in that same
    coordinate frame (data/processed/aligned_raw/masks/{patient_id}.png).
    Binarized to {0.0, 1.0} and shaped [1, H, W].

    Only 201 of 217 patients have a successful alignment (16 were rejected
    by build_aligned_dataset.py's geometric sanity checks and, per a
    deliberate decision, are excluded rather than manually annotated --
    see CLAUDE.md Sec 1.4.3). This class filters to those 201 by joining
    against alignment_log.csv, WITHOUT modifying dataset_splits.csv itself
    -- that CSV is shared with AnemiaClassificationDataset, for which all
    217 patients (including the 16) remain perfectly valid; only this
    class's input data is incomplete for them.

    tissue_type selects which of the two aligned datasets to read from --
    "palpebral" (default, unchanged behavior) or "forniceal_palpebral" (the
    wider combined fornix+palpebral view, aligned by the sibling script
    build_aligned_dataset_forniceal.py: 211/217 patients, 6 excluded for
    having no forniceal_palpebral crop at all, 0 genuine alignment
    failures). Explicit images_dir/masks_dir/alignment_log_csv arguments,
    if given, override whatever tissue_type would have resolved to."""

    def __init__(
        self,
        split: str,
        splits_csv: Path = SPLITS_CSV,
        tissue_type: str = "palpebral",
        images_dir: Path = None,
        masks_dir: Path = None,
        alignment_log_csv: Path = None,
        transform=None,
    ):
        if tissue_type not in ALIGNED_TISSUE_CONFIG:
            raise ValueError(
                f"Unknown tissue_type {tissue_type!r}; expected one of {list(ALIGNED_TISSUE_CONFIG)}"
            )
        tissue_config = ALIGNED_TISSUE_CONFIG[tissue_type]
        images_dir = images_dir if images_dir is not None else tissue_config["images_dir"]
        masks_dir = masks_dir if masks_dir is not None else tissue_config["masks_dir"]
        alignment_log_csv = (
            alignment_log_csv if alignment_log_csv is not None else tissue_config["alignment_log_csv"]
        )

        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split]

        alignment_log = pd.read_csv(alignment_log_csv)
        aligned_ids = set(alignment_log.loc[alignment_log["status"] == "ok", "patient_id"])
        df = df[df["patient_id"].isin(aligned_ids)]

        self.df = df.reset_index(drop=True)
        self.tissue_type = tissue_type
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        patient_id = self.df.loc[idx, "patient_id"]
        image = np.array(Image.open(self.images_dir / f"{patient_id}.jpg").convert("RGB"))
        mask = np.array(Image.open(self.masks_dir / f"{patient_id}.png").convert("L"))
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask


class AnemiaClassificationDataset(Dataset):
    """Returns (image, label). image is the full raw eye photo; label is
    the WHO-threshold anemic_label (0.0/1.0) from the metadata."""

    def __init__(self, split: str, splits_csv: Path = SPLITS_CSV, images_dir: Path = IMAGES_DIR, transform=None):
        df = pd.read_csv(splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.images_dir = Path(images_dir)
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

        return image, label


# --------------------------------------------------------------------------
# DataLoaders
# --------------------------------------------------------------------------
def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = 0) -> dict:
    train_tf = get_train_transforms()
    eval_tf = get_eval_transforms()

    datasets = {
        "aligned_seg_train": AlignedConjunctivaSegmentationDataset(split="train", transform=train_tf),
        "aligned_seg_val": AlignedConjunctivaSegmentationDataset(split="val", transform=eval_tf),
        "aligned_seg_test": AlignedConjunctivaSegmentationDataset(split="test", transform=eval_tf),
        "aligned_seg_forniceal_train": AlignedConjunctivaSegmentationDataset(
            split="train", tissue_type="forniceal_palpebral", transform=train_tf
        ),
        "aligned_seg_forniceal_val": AlignedConjunctivaSegmentationDataset(
            split="val", tissue_type="forniceal_palpebral", transform=eval_tf
        ),
        "aligned_seg_forniceal_test": AlignedConjunctivaSegmentationDataset(
            split="test", tissue_type="forniceal_palpebral", transform=eval_tf
        ),
        "cls_train": AnemiaClassificationDataset(split="train", transform=train_tf),
        "cls_val": AnemiaClassificationDataset(split="val", transform=eval_tf),
        "cls_test": AnemiaClassificationDataset(split="test", transform=eval_tf),
    }

    return {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=name.endswith("_train"),
            num_workers=num_workers,
        )
        for name, ds in datasets.items()
    }


# --------------------------------------------------------------------------
# Test block
# --------------------------------------------------------------------------
if __name__ == "__main__":
    create_patient_splits()
    loaders = get_dataloaders()

    aligned_images, aligned_masks = next(iter(loaders["aligned_seg_train"]))
    print("\n--- Aligned segmentation batch (palpebral) ---")
    print("image shape [B, C, H, W]:", tuple(aligned_images.shape))
    print("mask shape  [B, C, H, W]:", tuple(aligned_masks.shape))
    print("mask min/max:", aligned_masks.min().item(), aligned_masks.max().item())
    print("dataset sizes (train/val/test):",
          len(loaders["aligned_seg_train"].dataset),
          len(loaders["aligned_seg_val"].dataset),
          len(loaders["aligned_seg_test"].dataset))

    aligned_forniceal_images, aligned_forniceal_masks = next(iter(loaders["aligned_seg_forniceal_train"]))
    print("\n--- Aligned segmentation batch (forniceal_palpebral) ---")
    print("image shape [B, C, H, W]:", tuple(aligned_forniceal_images.shape))
    print("mask shape  [B, C, H, W]:", tuple(aligned_forniceal_masks.shape))
    print("mask min/max:", aligned_forniceal_masks.min().item(), aligned_forniceal_masks.max().item())
    print("dataset sizes (train/val/test):",
          len(loaders["aligned_seg_forniceal_train"].dataset),
          len(loaders["aligned_seg_forniceal_val"].dataset),
          len(loaders["aligned_seg_forniceal_test"].dataset))

    cls_images, cls_labels = next(iter(loaders["cls_train"]))
    print("\n--- Classification batch ---")
    print("image shape [B, C, H, W]:", tuple(cls_images.shape))
    print("label shape [B]:", tuple(cls_labels.shape))
    print("label values:", cls_labels.tolist())
