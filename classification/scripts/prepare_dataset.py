"""
Phase 4 (Classification): independent data preparation pipeline.

Deliberately NOT shared code with scripts/phase0_prepare_dataset.py -- this is
a from-scratch reimplementation against the same source archive, so that the
classification/ module has zero dependency on the segmentation phase's code
(only the same immutable source data, archive.zip, is shared). It solves the
same handful of real data-quality problems the source data has (verified
independently against the extracted raw archive, not assumed):

  - Hgb values are comma-decimal text for some rows ("15,1") and a literal
    "_" placeholder for at least one unrecorded case (Italy patient 93).
  - The "ELIMINATO" exclusion flag can appear in any column of the row, not
    just the core 5 (verified: Italy patient 93, column "Unnamed: 6").
  - Palpebral crop PNGs can have a corrupted CRC on their ancillary 'iCCP'
    chunk, which makes Pillow refuse to open the file until that chunk is
    stripped.
  - A PNG's "transparent" region is NOT already black at the source
    resolution (verified: mean RGB ~36 in the alpha<=127 region of
    India folder 1's palpebral.png) -- naive .convert("RGB") alone does not
    give a clean black background. An explicit Image.alpha_composite onto a
    solid black RGBA canvas is required to actually zero it out (verified:
    0.0 mean / 0 max after compositing, vs. ~36 without it).

Labels use WHO thresholds for non-pregnant adults (Male Hb < 13.0 g/dL,
Female Hb < 12.0 g/dL), same rule for both countries -- NOT the
country/gender-specific thresholds floated earlier in this session, which
were unsourced and would have widened the already-documented India/Italy
anemia-rate gap (a confound this project treats as a serious risk, see
CLAUDE.md Sec 0.5). The dataset has no pregnancy field, so all female
patients are assumed non-pregnant -- a stated limitation, not an oversight,
carried over unchanged from the root project's Sec 0.1.

Extracts BOTH the palpebral and forniceal_palpebral crops (two genuinely
different segmentations per the dataset's own documentation, Dataset
anemia.docx) for every retained patient. Six Italy folders documented as
missing the forniceal conjunctiva entirely (1, 35, 54, 58, 75, 109) have no
forniceal_palpebral file at all -- those patients are cleanly excluded from
the forniceal_palpebral variant only (not from palpebral), logged rather
than silently skipped, mirroring the root project's precedent for honest
exclusion over forcing/faking missing data (CLAUDE.md Sec 1.4.3).
"""

import io
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODULE_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = MODULE_ROOT / "data" / "raw" / "dataset anemia"
PROCESSED_DIR = MODULE_ROOT / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"

METADATA_CSV = PROCESSED_DIR / "metadata.csv"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"
EXTRACTION_LOG_CSV = PROCESSED_DIR / "extraction_log.csv"

COUNTRIES = {
    "India": RAW_ROOT / "India",
    "Italy": RAW_ROOT / "Italy",
}

TARGET_SIZE = 256
SEED = 42
TISSUE_TYPES = ["palpebral", "forniceal_palpebral"]

# WHO thresholds for non-pregnant adults, g/dL -- same rule both countries.
WHO_THRESHOLDS = {"M": 13.0, "F": 12.0}


# --------------------------------------------------------------------------
# Metadata loading
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Patient-level stratified splitting (4-way: country + label)
# --------------------------------------------------------------------------
def create_patient_splits(metadata: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    strata = metadata["country"] + "_" + metadata["anemic_label"].astype(int).astype(str)

    train_df, temp_df = train_test_split(
        metadata, test_size=0.30, stratify=strata, random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=strata.loc[temp_df.index], random_state=seed
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    result = pd.concat([train_df, val_df, test_df]).sort_index()
    return result[["patient_id", "country", "gender", "age", "hgb", "anemic_label", "split"]]


# --------------------------------------------------------------------------
# Image extraction
# --------------------------------------------------------------------------
CRITICAL_PNG_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}


def sanitize_png_bytes(data: bytes) -> bytes:
    """Strip ancillary PNG chunks with a corrupted CRC (verified: the source
    palpebral/forniceal_palpebral crops can have a bad CRC on 'iCCP', which
    makes Pillow refuse to open the file at all). A bad CRC on a critical
    chunk (actual pixel data) raises instead of being silently dropped."""
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


def flatten_to_black(rgba_img: Image.Image) -> Image.Image:
    """Explicit alpha-composite onto a solid black RGBA canvas, THEN drop
    the alpha channel. Verified necessary: the source crop's transparent
    region is not already black (mean RGB ~36 at native resolution) --
    plain .convert("RGB") would leave visible noise in the background.
    alpha_composite gives an exact, guaranteed 0.0 mean / 0 max result."""
    black_bg = Image.new("RGBA", rgba_img.size, (0, 0, 0, 255))
    return Image.alpha_composite(black_bg, rgba_img).convert("RGB")


def pad_to_square(img: Image.Image, fill=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new(img.mode, (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def find_crop_files(folder: Path) -> dict:
    """Classifies every PNG in a patient folder into palpebral /
    forniceal_palpebral / forniceal (the third, unused type) by filename.
    Returns {"palpebral": Path, "forniceal_palpebral": Path or None}.

    Typo-tolerant by construction, not by keyword-guessing: the "palpebral"
    spelling itself varies in the source data (verified: Italy folder 95
    uses "..._forniceal_palplebral.png" and "..._palplebral.png", an 'l'
    swap that a literal "forniceal_palpebral" substring match silently
    misses -- it would fall through and get skipped as a plain-forniceal
    file instead, silently dropping that patient's forniceal_palpebral crop
    with no error). Instead: the one PNG NOT containing "forniceal" at all
    is always the palpebral crop (Phase 0 verified this holds with zero
    exceptions across all 218 folders, regardless of "palpebral" spelling).
    Among PNGs that DO contain "forniceal", the combined view is always the
    longer filename (it's the forniceal-only name plus a palpebral-ish
    suffix, however spelled) -- robust to spelling, not to file count, so an
    unexpected count (not 0 or 2) raises rather than guessing."""
    pngs = [p for p in folder.iterdir() if p.suffix.lower() == ".png" and "(1)" not in p.name]

    non_forniceal = [p for p in pngs if "forniceal" not in p.name.lower()]
    forniceal_related = [p for p in pngs if "forniceal" in p.name.lower()]

    if len(non_forniceal) != 1:
        raise ValueError(f"{folder}: expected exactly 1 non-forniceal (palpebral) png, found {[p.name for p in non_forniceal]}")

    result = {"palpebral": non_forniceal[0], "forniceal_palpebral": None}

    if len(forniceal_related) == 0:
        pass  # documented case: forniceal conjunctiva not exposed in this photo
    elif len(forniceal_related) == 2:
        result["forniceal_palpebral"] = max(forniceal_related, key=lambda p: len(p.name))
    else:
        raise ValueError(
            f"{folder}: expected 0 or 2 forniceal-related pngs, found {[p.name for p in forniceal_related]}"
        )

    return result


def process_crop(src_path: Path) -> Image.Image:
    with open(src_path, "rb") as f:
        raw = f.read()
    img = Image.open(io.BytesIO(sanitize_png_bytes(raw))).convert("RGBA")
    flat = flatten_to_black(img)
    flat = pad_to_square(flat, fill=(0, 0, 0))
    flat = flat.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return flat


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    for tissue in TISSUE_TYPES:
        (IMAGES_DIR / tissue).mkdir(parents=True, exist_ok=True)

    frames = [load_country_metadata(country, path) for country, path in COUNTRIES.items()]
    meta = pd.concat(frames, ignore_index=True)
    meta["anemic_label"] = meta.apply(compute_anemic_label, axis=1)

    included = meta[~meta["excluded"]].copy()
    excluded = meta[meta["excluded"]]
    print(f"Total patients in metadata: {len(meta)}")
    print(f"Excluded: {len(excluded)}")
    if len(excluded):
        print(excluded[["patient_id", "exclusion_reason"]].to_string(index=False))

    included.to_csv(METADATA_CSV, index=False)
    print(f"\nSaved {len(included)}-patient metadata to {METADATA_CSV}")
    print("\n--- Class balance (WHO thresholds) ---")
    print(included["anemic_label"].value_counts())
    print("\n--- Anemia rate by country ---")
    print(included.groupby("country")["anemic_label"].agg(["count", "mean"]))

    splits = create_patient_splits(included)
    splits.to_csv(SPLITS_CSV, index=False)
    print(f"\nSaved splits to {SPLITS_CSV}")
    print(splits.groupby(["split", "country"])["anemic_label"].agg(["count", "mean"]))

    log_rows = []
    for _, row in included.iterrows():
        patient_id, country, number = row["patient_id"], row["country"], row["number"]
        folder = COUNTRIES[country] / str(number)
        crop_files = find_crop_files(folder)

        entry = {"patient_id": patient_id, "country": country}
        for tissue in TISSUE_TYPES:
            src = crop_files[tissue]
            if src is None:
                entry[f"{tissue}_status"] = "missing_source_file"
                continue
            out_img = process_crop(src)
            out_path = IMAGES_DIR / tissue / f"{patient_id}.jpg"
            out_img.save(out_path, quality=95)
            entry[f"{tissue}_status"] = "ok"
        log_rows.append(entry)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(EXTRACTION_LOG_CSV, index=False)
    print(f"\nSaved extraction log to {EXTRACTION_LOG_CSV}")
    for tissue in TISSUE_TYPES:
        counts = log_df[f"{tissue}_status"].value_counts()
        print(f"\n--- {tissue} extraction status ---")
        print(counts)


if __name__ == "__main__":
    main()
