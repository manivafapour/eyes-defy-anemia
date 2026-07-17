"""
Phase 1: Dual Data Pipeline Construction for the Eyes-Defy-Anemia project.

Builds a patient-level, country+label-stratified train/val/test split on top
of the Phase 0 metadata, then exposes three PyTorch datasets:

- ConjunctivaSegmentationDataset: (image, mask) pairs from
  data/processed/masks/{patient_id}_palpebral.png. The RGB channels are the
  image and the alpha channel (binarized) is the mask -- they are the only
  pixel-aligned pair available from Phase 0 alone, since the raw eye photo
  (data/processed/images) and the palpebral crop went through independent
  crop/pad/resize steps in Phase 0 and do not share a coordinate grid.
- AlignedConjunctivaSegmentationDataset: (image, mask) pairs from
  data/processed/aligned_raw/{images,masks}/, built by
  scripts/build_aligned_dataset.py via template-matching-based alignment.
  image is the FULL raw photo; mask is a genuinely pixel-aligned tissue mask
  in that same coordinate frame -- the dataset to use for a segmentation
  model intended to generalize to raw photos (see CLAUDE.md Sec 1.4).
- AnemiaClassificationDataset: (image, label) pairs from
  data/processed/images/{patient_id}.jpg and the metadata's anemic_label.
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
MASKS_DIR = PROCESSED_DIR / "masks"

ALIGNED_ROOT = PROCESSED_DIR / "aligned_raw"
ALIGNED_IMAGES_DIR = ALIGNED_ROOT / "images"
ALIGNED_MASKS_DIR = ALIGNED_ROOT / "masks"

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
class ConjunctivaSegmentationDataset(Dataset):
    """Returns (image, mask). image is the palpebral crop's RGB channels;
    mask is that same file's alpha channel, binarized to {0.0, 1.0} and
    shaped [1, H, W]."""

    def __init__(self, split: str, splits_csv: Path = SPLITS_CSV, masks_dir: Path = MASKS_DIR, transform=None):
        df = pd.read_csv(splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        patient_id = self.df.loc[idx, "patient_id"]
        rgba = np.array(Image.open(self.masks_dir / f"{patient_id}_palpebral.png").convert("RGBA"))
        image = rgba[..., :3]
        mask = (rgba[..., 3] > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask


class AlignedConjunctivaSegmentationDataset(Dataset):
    """Returns (image, mask) from the raw-photo-aligned dataset
    (data/processed/aligned_raw/, built by scripts/build_aligned_dataset.py).
    Unlike ConjunctivaSegmentationDataset, image is the FULL raw clinical
    photo (data/processed/aligned_raw/images/{patient_id}.jpg) and mask is a
    genuinely pixel-aligned tissue mask in that same coordinate frame
    (data/processed/aligned_raw/masks/{patient_id}.png), recovered via
    template matching rather than sharing a single source file. Binarized
    to {0.0, 1.0} and shaped [1, H, W], same as ConjunctivaSegmentationDataset."""

    def __init__(
        self,
        split: str,
        splits_csv: Path = SPLITS_CSV,
        images_dir: Path = ALIGNED_IMAGES_DIR,
        masks_dir: Path = ALIGNED_MASKS_DIR,
        transform=None,
    ):
        df = pd.read_csv(splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
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
        "seg_train": ConjunctivaSegmentationDataset(split="train", transform=train_tf),
        "seg_val": ConjunctivaSegmentationDataset(split="val", transform=eval_tf),
        "seg_test": ConjunctivaSegmentationDataset(split="test", transform=eval_tf),
        "aligned_seg_train": AlignedConjunctivaSegmentationDataset(split="train", transform=train_tf),
        "aligned_seg_val": AlignedConjunctivaSegmentationDataset(split="val", transform=eval_tf),
        "aligned_seg_test": AlignedConjunctivaSegmentationDataset(split="test", transform=eval_tf),
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

    seg_images, seg_masks = next(iter(loaders["seg_train"]))
    print("\n--- Segmentation batch ---")
    print("image shape [B, C, H, W]:", tuple(seg_images.shape))
    print("mask shape  [B, C, H, W]:", tuple(seg_masks.shape))
    print("mask min/max:", seg_masks.min().item(), seg_masks.max().item())

    aligned_images, aligned_masks = next(iter(loaders["aligned_seg_train"]))
    print("\n--- Aligned segmentation batch ---")
    print("image shape [B, C, H, W]:", tuple(aligned_images.shape))
    print("mask shape  [B, C, H, W]:", tuple(aligned_masks.shape))
    print("mask min/max:", aligned_masks.min().item(), aligned_masks.max().item())

    cls_images, cls_labels = next(iter(loaders["cls_train"]))
    print("\n--- Classification batch ---")
    print("image shape [B, C, H, W]:", tuple(cls_images.shape))
    print("label shape [B]:", tuple(cls_labels.shape))
    print("label values:", cls_labels.tolist())
