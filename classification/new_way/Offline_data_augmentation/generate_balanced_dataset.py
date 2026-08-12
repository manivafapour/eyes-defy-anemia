"""
Offline, country-stratified label balancing for the new_way/ roster's
TRAIN+VAL pool (the K-fold CV pool -- see new_way/kfold/kfold_engine.py;
TEST is never touched by this script -- rebalancing eval/test data would
invalidate every metric computed against it).

Widened 2026-08-12 from "train split only" (151/147 patients) to "train+val
pooled" (184/178) specifically to support the 3-fold retraining of the 6
best new_way combos: every K-fold CV precedent in this repo pools train+val
patients before folding (classification/step1_cv_harness/cv_data.py,
datapreparepipeline/efficientnet_b0_forniceal_5fold_cv/cv_dataset.py,
Segmentation/scripts/train_pretrained_kfold/kfold_engine.py) so a manifest
built over "train" alone would silently have zero entries for any patient
whose real split is "val" -- exactly the patients a fold's training
partition needs to be able to include. This is a pure widening (train+val
is a superset of train), so it doesn't invalidate anything already built
against the old, narrower manifest -- it was never used for real training.

Why country-stratified, not a plain overall label balance: the real pool
imbalance is much sharper WITHIN each country than overall --

    India:  non-anemic=23, anemic=57  (2.5x)
    Italy:  non-anemic=84, anemic=20  (4.2x)  [palpebral pool]
    Overall: non-anemic=107, anemic=77 (1.4x -- looks mild, hides the above)

-- and this project's own memory (classification/.project_memory/
01_roadmap.md, 07_step1_measurement_harness.md) repeatedly documents models
exploiting exactly this country<->label correlation as a shortcut ("India-
anemic vs Italy-healthy" pairs driving overall AUC; pos_weight biasing
toward "predict anemic" differently per country). Balancing each country to
its own 50/50 label split removes that shortcut's signal at the source
(P(anemic | India) = P(anemic | Italy) = 0.5 in the resulting train set)
without inflating either country beyond what's needed for that -- confirmed
with the project author as the deliberate design choice (2026-08-12),
over a plain overall-label balance or a full 4-cell equalization to the
single largest cell (which would over-augment India beyond what's needed).

Method: for each (tissue_type, country), the minority label's real patients
are oversampled via deterministic, geometric-only augmentation (horizontal
flip + a fixed rotation angle -- no color/brightness jitter, matching this
project's established, literature-backed convention of zero color
augmentation, 04_literature_review_findings.md) until that label's count
matches the country's majority label. Majority-label patients, and the
minority label's own real patients, are copied through unchanged (byte-
identical file copy, not re-encoded). Six fixed recipes are cycled
round-robin (sorted patient_id order) so no patient ever receives the same
recipe twice for one tissue type; the two real worst-case ratios on the
train+val pool (Italy palpebral: deficit=64 over 20 patients, needs 4
rounds; Italy forniceal_palpebral: deficit=60 over 19, needs 4 rounds)
comfortably fit inside the 6-recipe pool with no repeats.

Run: python generate_balanced_dataset.py
(safe to re-run -- output images/manifest are fully regenerated each time,
same convention as classification/datapreparepipeline/prepare_dataset.py)
"""

import shutil
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
NEW_WAY_DIR = Path(__file__).resolve().parent.parent
CLASSIFICATION_DIR = NEW_WAY_DIR.parent
SOURCE_PROCESSED_DIR = CLASSIFICATION_DIR / "data" / "processed"
SOURCE_IMAGES_DIR = SOURCE_PROCESSED_DIR / "images"
SPLITS_CSV = SOURCE_PROCESSED_DIR / "splits.csv"
EXTRACTION_LOG_CSV = SOURCE_PROCESSED_DIR / "extraction_log.csv"

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"

TISSUE_TYPES = ["palpebral", "forniceal_palpebral"]
COUNTRIES = ["India", "Italy"]

# (recipe_name, horizontal_flip, rotation_degrees) -- deterministic, exact
# angle (not a random range) so every call to the same recipe on the same
# source image produces the identical output, matching this project's
# "verify empirically, reproducible" convention. Angles stay within the
# existing online-augmentation bound (Rotate(limit=15), dataset.py) except
# for two slightly wider ones (used only in later rounds, when they exist)
# to keep all 6 recipes visually distinct from each other.
RECIPES = [
    ("r1_rot+10", False, 10),
    ("r2_flip_rot-10", True, -10),
    ("r3_rot-15", False, -15),
    ("r4_flip_rot+15", True, 15),
    ("r5_rot+7", False, 7),
    ("r6_flip_rot-7", True, -7),
]


def apply_recipe(image: np.ndarray, hflip: bool, angle: float) -> np.ndarray:
    ops = []
    if hflip:
        ops.append(A.HorizontalFlip(p=1.0))
    ops.append(A.Rotate(limit=(angle, angle), border_mode=0, fill=0, p=1.0))
    transform = A.Compose(ops)
    return transform(image=image)["image"]


def load_source_image(tissue_type: str, patient_id: str) -> np.ndarray:
    path = SOURCE_IMAGES_DIR / tissue_type / f"{patient_id}.jpg"
    return np.array(Image.open(path).convert("RGB"))


def save_image(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image).save(path, quality=95)


def copy_original(tissue_type: str, patient_id: str, out_dir: Path) -> None:
    src = SOURCE_IMAGES_DIR / tissue_type / f"{patient_id}.jpg"
    shutil.copy2(src, out_dir / f"{patient_id}.jpg")


# --------------------------------------------------------------------------
# Per-(tissue_type, country) balancing
# --------------------------------------------------------------------------
def balance_country_group(df_country: pd.DataFrame, tissue_type: str, country: str, out_dir: Path) -> list[dict]:
    rows = []
    counts = df_country["anemic_label"].value_counts()
    majority_label = float(counts.idxmax())
    minority_label = float(counts.idxmin())
    majority_df = df_country[df_country["anemic_label"] == majority_label]
    minority_df = df_country[df_country["anemic_label"] == minority_label]
    deficit = len(majority_df) - len(minority_df)

    # Both labels' real patients are copied through unchanged first.
    for _, row in pd.concat([majority_df, minority_df]).iterrows():
        pid = row["patient_id"]
        copy_original(tissue_type, pid, out_dir)
        rows.append(
            {
                "image_id": pid,
                "source_patient_id": pid,
                "tissue_type": tissue_type,
                "country": country,
                "anemic_label": row["anemic_label"],
                "is_augmented": False,
                "recipe": "original",
            }
        )

    if deficit == 0:
        print(f"  [{tissue_type}/{country}] already balanced ({len(majority_df)}/{len(minority_df)}), no augmentation needed")
        return rows

    minority_ids = sorted(minority_df["patient_id"].tolist())
    n_minority = len(minority_ids)
    max_rounds_available = len(RECIPES)
    rounds_needed = -(-deficit // n_minority)  # ceil division
    if rounds_needed > max_rounds_available:
        print(
            f"  [{tissue_type}/{country}] WARNING: deficit={deficit} over {n_minority} minority patients "
            f"needs {rounds_needed} rounds, only {max_rounds_available} distinct recipes available -- "
            f"some (patient, recipe) pairs will repeat."
        )

    generated = 0
    round_idx = 0
    while generated < deficit:
        recipe_name, hflip, angle = RECIPES[round_idx % len(RECIPES)]
        remaining = deficit - generated
        batch = minority_ids[: min(n_minority, remaining)]
        for pid in batch:
            image = load_source_image(tissue_type, pid)
            augmented = apply_recipe(image, hflip, angle)
            aug_id = f"{pid}_aug{round_idx + 1}"
            save_image(augmented, out_dir / f"{aug_id}.jpg")
            label = minority_df.loc[minority_df["patient_id"] == pid, "anemic_label"].iloc[0]
            rows.append(
                {
                    "image_id": aug_id,
                    "source_patient_id": pid,
                    "tissue_type": tissue_type,
                    "country": country,
                    "anemic_label": label,
                    "is_augmented": True,
                    "recipe": recipe_name,
                }
            )
            generated += 1
            if generated >= deficit:
                break
        round_idx += 1

    print(
        f"  [{tissue_type}/{country}] majority(label={majority_label:.0f})={len(majority_df)}, "
        f"minority(label={minority_label:.0f})={len(minority_df)} -> generated {deficit} augmented "
        f"-> balanced at {len(majority_df)}/{len(majority_df)}"
    )
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    splits = pd.read_csv(SPLITS_CSV)
    log = pd.read_csv(EXTRACTION_LOG_CSV)
    pool = splits[splits["split"].isin(["train", "val"])]  # K-fold CV pool; test stays sealed

    all_rows = []
    for tissue_type in TISSUE_TYPES:
        out_dir = OUTPUT_IMAGES_DIR / tissue_type
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ok_ids = set(log.loc[log[f"{tissue_type}_status"] == "ok", "patient_id"])
        df_tissue = pool[pool["patient_id"].isin(ok_ids)]

        print(f"\n=== {tissue_type} (train+val pool: {len(df_tissue)} patients) ===")
        for country in COUNTRIES:
            df_country = df_tissue[df_tissue["country"] == country]
            rows = balance_country_group(df_country, tissue_type, country, out_dir)
            all_rows.extend(rows)

    manifest = pd.DataFrame(all_rows)
    manifest.to_csv(MANIFEST_CSV, index=False)

    print(f"\nSaved manifest ({len(manifest)} rows) to {MANIFEST_CSV}")
    print("\nFinal balanced counts (tissue_type x country x anemic_label):")
    print(manifest.groupby(["tissue_type", "country", "anemic_label"]).size())
    print("\nOriginal vs. augmented image counts per tissue_type:")
    print(manifest.groupby(["tissue_type", "is_augmented"]).size())


if __name__ == "__main__":
    main()
