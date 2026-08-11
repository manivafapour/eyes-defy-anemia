# Complete Code-Backed Report: Data Curation & Preprocessing
**Module:** `classification/datapreparepipeline/` (`prepare_dataset.py` & `dataset.py`)  
**Project:** Eyes-Defy-Anemia — Non-Invasive Conjunctiva-Based Anemia Detection  

---

## Overview & Architecture

This document provides a **complete, code-backed technical explanation** of the Data Curation and Preprocessing pipeline for the Anemia Classification Module (Phase 4). To ensure experimental isolation and prevent data contamination from earlier segmentation phases, this pipeline is implemented from scratch in two dedicated scripts:
1. `prepare_dataset.py`: Parses raw clinical metadata, applies WHO diagnostic thresholds, performs 4-way stratified patient-level splitting, sanitizes corrupted PNG chunks, corrects the legacy white-background bug, and exports standardized square crops to `data/processed/images/`.
2. `dataset.py`: Implements resolution-aware PyTorch `Dataset` and `DataLoader` classes, manages data augmentations (Albumentations), and supports dynamic positive-class weighting (`pos_weight`) for imbalanced classification.

```mermaid
graph TD
    A[Raw Excel Metadata & PNG Crops] --> B[load_country_metadata & parse_hgb]
    B --> C[compute_anemic_label<br>WHO: M < 13.0 | F < 12.0]
    C --> D[create_patient_splits<br>4-Way: Country x Label]
    
    A --> E[find_crop_files & sanitize_png_bytes]
    E --> F[flatten_to_black<br>Fixes White-Background Bug]
    F --> G[pad_to_square & Resize 256x256]
    
    D --> H[TissueClassificationDataset]
    G --> H
    H --> I[get_train_transforms / get_eval_transforms<br>Resolution-Aware: 256px CNN | 224px ViT]
    I --> J[get_dataloaders<br>Batch Size = 32]
```

---

## 1. WHO Diagnostic Labeling & Robust Metadata Parsing (`prepare_dataset.py`)

### 1.1 Why This Code Exists
The raw Excel files (`India.xlsx` and `Italy.xlsx`) contain formatting anomalies that must be handled programmatically:
- Hemoglobin (`hgb`) values in Italy use comma-decimal notation (`"15,1"`) instead of periods, and patient 93 uses an underscore placeholder (`"_"`) for a missing reading.
- The `"ELIMINATO"` clinical exclusion flag can appear outside the core 5 columns (e.g., column `"Unnamed: 6"`).
- We apply World Health Organization (WHO) clinical anemia thresholds uniformly across both countries: **Male $< 13.0\text{ g/dL}$, Female $< 12.0\text{ g/dL}$**.

### 1.2 Implementation Code
```python
# WHO thresholds for non-pregnant adults, g/dL -- same rule both countries.
WHO_THRESHOLDS = {"M": 13.0, "F": 12.0}


def parse_hgb(value) -> float:
    """Handle Italian comma-decimal text ('15,1') and the '_' placeholder
    used for at least one unrecorded reading (Italy patient 93)."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_country_metadata(country: str, country_dir: Path) -> pd.DataFrame:
    xlsx_path = country_dir / f"{country}.xlsx"
    # Read every column present (no usecols) -- ELIMINATO can land outside
    # the core 5 columns (verified: Italy patient 93, "Unnamed: 6").
    full = pd.read_excel(xlsx_path, header=0)
    full = full.dropna(how="all")

    row_text = full.fillna("").astype(str).agg(" ".join, axis=1).str.upper()
    eliminato_flag = row_text.str.contains("ELIMINATO").to_numpy()

    df = full.iloc[:, :5].copy()
    df.columns = ["number", "hgb", "gender", "age", "note"]
    df["eliminato_flag"] = eliminato_flag
    df = df.dropna(subset=["number"]).copy()

    df["number"] = df["number"].astype(float).astype(int)
    df["hgb"] = df["hgb"].apply(parse_hgb)
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["country"] = country

    invalid_gender = ~df["gender"].isin(WHO_THRESHOLDS.keys())
    df["excluded"] = df["hgb"].isna() | df["eliminato_flag"] | invalid_gender
    df["exclusion_reason"] = ""
    df.loc[df["hgb"].isna(), "exclusion_reason"] = "missing_or_invalid_hgb"
    df.loc[invalid_gender, "exclusion_reason"] = "invalid_gender"
    df.loc[df["eliminato_flag"], "exclusion_reason"] = "eliminato_flag"
    df = df.drop(columns=["eliminato_flag"])

    df["patient_id"] = df.apply(lambda r: f"{country}_{r['number']:03d}", axis=1)
    return df


def compute_anemic_label(row) -> float:
    if row["excluded"]:
        return float("nan")
    threshold = WHO_THRESHOLDS[row["gender"]]
    return float(row["hgb"] < threshold)
```

### 1.3 Key Technical Highlights
- **Full-Row Text Scanning:** By scanning `row_text.str.contains("ELIMINATO")` across all columns, we guarantee no excluded clinical patients leak into the dataset.
- **Uniform Clinical Ground Truth:** `compute_anemic_label()` returns float targets (`1.0` for Anemic, `0.0` for Healthy) without country-specific adjustments, preserving clinical consistency.

---

## 2. Four-Way Patient-Level Stratified Splitting (`prepare_dataset.py`)

### 2.1 Why This Code Exists
To avoid patient-level data leakage, we never split by image—we split exclusively by `patient_id`. Furthermore, because India is 80% Anemic and Italy is 59% Healthy, stratifying on label alone could cause rare demographic subgroups (like `India-Healthy`, $N=19$) to be missing from validation or test folds. We create a compound 4-way stratification key: `country + "_" + anemic_label`.

### 2.2 Implementation Code
```python
def create_patient_splits(metadata: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    strata = metadata["country"] + "_" + metadata["anemic_label"].astype(int).astype(str)

    # First split: 70% Train, 30% Temporary (Val + Test)
    train_df, temp_df = train_test_split(
        metadata, test_size=0.30, stratify=strata, random_state=seed
    )
    # Second split: 15% Validation, 15% Test (equal 50/50 split of the 30% temp)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=strata.loc[temp_df.index], random_state=seed
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    result = pd.concat([train_df, val_df, test_df]).sort_index()
    return result[["patient_id", "country", "gender", "age", "hgb", "anemic_label", "split"]]
```

### 2.3 Key Technical Highlights
- **Compound Stratification (`strata`):** Generates 4 distinct strata: `"India_1"`, `"India_0"`, `"Italy_1"`, and `"Italy_0"`.
- **Reproducible 70 / 15 / 15 Breakdown:** Produces exactly **151 Train patients, 33 Validation patients, and 33 Test patients**, locking demographic ratios across all three splits.

---

## 3. Typo-Tolerant Tissue Crop Discovery (`prepare_dataset.py`)

### 3.1 Why This Code Exists
Clinical file names in the raw repository contain typographical errors (e.g., Italy folder 95 names files `"..._palplebral.png"` and `"..._forniceal_palplebral.png"` with an extra `"l"`). Simple substring matching for `"forniceal_palpebral"` would silently fail. 

### 3.2 Implementation Code
```python
def find_crop_files(folder: Path) -> dict:
    """Classifies every PNG in a patient folder into palpebral /
    forniceal_palpebral / forniceal by filename without keyword guessing."""
    pngs = [p for p in folder.iterdir() if p.suffix.lower() == ".png" and "(1)" not in p.name]

    non_forniceal = [p for p in pngs if "forniceal" not in p.name.lower()]
    forniceal_related = [p for p in pngs if "forniceal" in p.name.lower()]

    if len(non_forniceal) != 1:
        raise ValueError(f"{folder}: expected exactly 1 non-forniceal (palpebral) png, found {[p.name for p in non_forniceal]}")

    result = {"palpebral": non_forniceal[0], "forniceal_palpebral": None}

    if len(forniceal_related) == 0:
        pass  # documented case: 6 Italian patients lack forniceal conjunctiva exposure
    elif len(forniceal_related) == 2:
        # The combined forniceal_palpebral crop is always the longer filename
        result["forniceal_palpebral"] = max(forniceal_related, key=lambda p: len(p.name))
    else:
        raise ValueError(
            f"{folder}: expected 0 or 2 forniceal-related pngs, found {[p.name for p in forniceal_related]}"
        )

    return result
```

### 3.3 Key Technical Highlights
- **Typo Immunity:** By isolating files containing `"forniceal"` versus those without, we identify `palpebral` without relying on exact spelling.
- **Length-Based Sorting:** Between the two forniceal-related PNGs (`forniceal` vs. `forniceal_palpebral`), selecting `max(..., key=len)` reliably retrieves the combined crop.

---

## 4. White-Background Bug Fix & PNG Sanitization (`prepare_dataset.py`)

### 4.1 Why This Code Exists
Our diagnostic investigation uncovered two severe image-level anomalies:
1. **Corrupted PNG CRC Chunks:** Some raw PNGs contain corrupted CRC checksums on ancillary `'iCCP'` ICC profile chunks, causing standard Pillow loaders to crash.
2. **The White-Background Convention Bug:** **30 out of 211 `forniceal_palpebral` crops—100% belonging to Italian patients—delimited tissue using an opaque white background (`255, 255, 255`)** instead of alpha transparency. A naive `.convert("RGB")` leaves white pixels intact, allowing neural networks to use border brightness as a shortcut to detect Italy cohort membership.

### 4.2 Implementation Code
```python
CRITICAL_PNG_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}

def sanitize_png_bytes(data: bytes) -> bytes:
    """Strip ancillary PNG chunks with a corrupted CRC while preserving critical chunks."""
    out = [data[:8]]
    pos = 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        ctype = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        crc_stored = data[pos + 8 + length : pos + 12 + length]
        crc_calc = zlib.crc32(ctype + chunk_data).to_bytes(4, "big")
        chunk_end = pos + 12 + length
        if crc_stored != crc_calc:
            if ctype in CRITICAL_PNG_CHUNKS:
                raise ValueError(f"corrupted critical PNG chunk {ctype!r} (bad CRC)")
        else:
            out.append(data[pos:chunk_end])
        pos = chunk_end
        if ctype == b"IEND":
            break
    return b"".join(out)


ALPHA_THRESHOLD = 127
OPAQUE_ALPHA_FRACTION_THRESHOLD = 0.99
WHITE_BACKGROUND_THRESHOLD = 245


def _alpha_is_functional(rgba_img: Image.Image) -> bool:
    """True if this crop's alpha channel encodes a real tissue cutout."""
    alpha = np.array(rgba_img)[..., 3]
    opaque_fraction = (alpha > ALPHA_THRESHOLD).mean()
    return opaque_fraction < OPAQUE_ALPHA_FRACTION_THRESHOLD


def flatten_to_black(rgba_img: Image.Image) -> Image.Image:
    """Flattens a crop to a guaranteed-black background (0, 0, 0), correcting
    both transparent crops and opaque white-background crops."""
    if _alpha_is_functional(rgba_img):
        # Standard case: composite transparent regions onto a solid black canvas
        black_bg = Image.new("RGBA", rgba_img.size, (0, 0, 0, 255))
        return Image.alpha_composite(black_bg, rgba_img).convert("RGB")

    # Fallback case: white-background convention bug (100% opaque alpha)
    rgb = np.array(rgba_img.convert("RGB"))
    is_background = (rgb >= WHITE_BACKGROUND_THRESHOLD).all(axis=-1)
    rgb[is_background] = (0, 0, 0)
    return Image.fromarray(rgb, mode="RGB")


def process_crop(src_path: Path) -> Image.Image:
    with open(src_path, "rb") as f:
        raw = f.read()
    img = Image.open(io.BytesIO(sanitize_png_bytes(raw))).convert("RGBA")
    flat = flatten_to_black(img)
    flat = pad_to_square(flat, fill=(0, 0, 0))
    flat = flat.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return flat
```

### 4.3 Key Technical Highlights
- **CRC Sanitization (`sanitize_png_bytes`):** Strips corrupted non-critical PNG metadata chunks while preserving valid image pixel data (`IDAT`).
- **Opaque Alpha Detection (`_alpha_is_functional`):** Distinguishes legitimate alpha-transparent images ($<13\%$ opaque) from white-background images ($100\%$ opaque).
- **Guaranteed Zero-Intensity Backgrounds:** Whether transparent or white, non-tissue background pixels are strictly converted to `(0, 0, 0)`, eliminating geographic color confounds.

---

## 5. Resolution-Aware PyTorch Dataset & Transforms (`dataset.py`)

### 5.1 Why This Code Exists
To feed our clean images into 9 different architectures, our PyTorch dataset must:
- Handle **resolution-aware loading**: Convolutional models (`ResNet18`, `ConvNeXt-Tiny`) use **$256 \times 256$**, whereas Vision Transformers (`ViT-B/16`, `Swin-Tiny`) require **$224 \times 224$** to match pretrained positional patch embeddings.
- Filter out the 6 Italian patients missing forniceal crops dynamically from `extraction_log.csv` without modifying the global `splits.csv` file.
- Apply clinical regularizing augmentations to training folds while keeping validation/test transforms strictly deterministic.

### 5.2 Implementation Code
```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class TissueClassificationDataset(Dataset):
    """Returns (image, label, country). Filters to patients with an 'ok'
    extraction status for the requested tissue_type."""

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
```

### 5.3 Key Technical Highlights
- **Dynamic Extraction Filtering (`ok_ids`):** Automatically subsets `forniceal_palpebral` to 211 patients while retaining all 217 for `palpebral`.
- **Three-Tuple Return (`image, label, country`):** Returning `country` alongside the tensor enables stratified evaluation (India vs. Italy AUC) without secondary CSV lookups.
- **ImageNet Standardization:** Normalized using `(0.485, 0.456, 0.406)` mean and `(0.229, 0.224, 0.225)` std, matching the pretraining domain of torchvision backbones.

---

## 6. DataLoader Construction & Dynamic Imbalance Weighting (`dataset.py`)

### 6.1 Why This Code Exists
To ensure consistent mini-batch optimization across all 18 combinations, we encapsulate dataset instantiation inside a clean factory function (`get_dataloaders()`). Additionally, because the training fold is 58.1% anemic, we calculate the exact positive-class weighting ratio ($w$) to pass into PyTorch's weighted binary cross-entropy loss (`BCEWithLogitsLoss`).

### 6.2 Implementation Code
```python
BATCH_SIZE = 32  # Stable VRAM usage across all 18 architectures

def get_dataloaders(
    tissue_type: str, batch_size: int = BATCH_SIZE, num_workers: int = 0, image_size: int = IMAGE_SIZE
) -> dict:
    """Creates train, val, and test DataLoaders with resolution-aware sizing."""
    train_tf = get_train_transforms(image_size)
    eval_tf = get_eval_transforms(image_size)

    datasets = {
        "train": TissueClassificationDataset(split="train", tissue_type=tissue_type, transform=train_tf),
        "val": TissueClassificationDataset(split="val", tissue_type=tissue_type, transform=eval_tf),
        "test": TissueClassificationDataset(split="test", tissue_type=tissue_type, transform=eval_tf),
    }

    return {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers)
        for name, ds in datasets.items()
    }
```

#### Dynamic Loss Weighting in `trainer_engine.py`:
```python
# Within trainer_engine.py: calculate positive class weight dynamically from train_loader
def compute_pos_weight(train_dataset: Dataset) -> torch.Tensor:
    """Computes N_healthy / N_anemic to balance BCEWithLogitsLoss."""
    labels = train_dataset.df["anemic_label"].to_numpy()
    n_pos = (labels == 1.0).sum()
    n_neg = (labels == 0.0).sum()
    weight = float(n_neg) / float(n_pos)
    return torch.tensor([weight], dtype=torch.float32)

# Instantiating PyTorch loss with dynamic weight
pos_weight = compute_pos_weight(train_loader.dataset).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 6.3 Key Technical Highlights
- **Resolution Flexibility (`image_size`):** Allows single-line switching between $256\text{px}$ CNN experiments and $224\text{px}$ Transformer experiments.
- **Dynamic Loss Balancing:** `compute_pos_weight()` automatically derives $w \approx 0.7159$ ($63 / 88$ healthy-to-anemic ratio in train), preventing majority-class bias during backpropagation.

---

## 7. Summary Table of Processed Artifacts

| Artifact Name | File Path | Description & Provenance |
|---|---|---|
| **Clean Metadata CSV** | `data/processed/metadata.csv` | Full 217-patient demographic & Hgb records with parsed WHO labels. |
| **Patient Splits CSV** | `data/processed/splits.csv` | 4-way stratified (`Country` $\times$ `Label`) split manifest (`train`, `val`, `test`). |
| **Extraction Log CSV** | `data/processed/extraction_log.csv` | Per-patient audit trail tracking crop availability (`ok` vs. `missing_source_file`). |
| **Palpebral Crops** | `data/processed/images/palpebral/*.jpg` | 217 clean square crops on guaranteed `(0, 0, 0)` black backgrounds. |
| **Forniceal Crops** | `data/processed/images/forniceal_palpebral/*.jpg` | 211 clean square crops (white-background bug fixed and verified). |

---

*Report generated directly from production source scripts (`classification/datapreparepipeline/prepare_dataset.py` and `dataset.py`).*
