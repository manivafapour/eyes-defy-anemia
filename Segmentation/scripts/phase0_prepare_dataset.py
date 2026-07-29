"""
Phase 0: Dataset Standardization for the Eyes-Defy-Anemia pipeline.

Reads directly from archive.zip (India + Italy folders), builds a unified,
cleaned metadata table with WHO-threshold anemia labels, and extracts the
raw eye photo + palpebral conjunctiva crop for every valid patient into a
resized, aspect-ratio-preserving square working dataset.

Anemia labels use WHO thresholds for non-pregnant adults (Male Hb < 13.0
g/dL, Female Hb < 12.0 g/dL). The dataset has no pregnancy field, so all
female patients are assumed non-pregnant -- a stated limitation, not an
oversight.
"""

import io
import zipfile
import zlib
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "archive.zip"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = OUTPUT_DIR / "images"
MASKS_DIR = OUTPUT_DIR / "masks"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"

TARGET_SIZE = 256
ZIP_ROOT = "dataset anemia"
COUNTRIES = {
    "India": f"{ZIP_ROOT}/India/India.xlsx",
    "Italy": f"{ZIP_ROOT}/Italy/Italy.xlsx",
}

WHO_THRESHOLDS = {"M": 13.0, "F": 12.0}  # g/dL, non-pregnant adult


# --------------------------------------------------------------------------
# Metadata loading
# --------------------------------------------------------------------------
def parse_hgb(value) -> float:
    """Handle Italian comma-decimal text values (e.g. '15,1') and the '_' placeholder."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_country_metadata(zf: zipfile.ZipFile, country: str, xlsx_path: str) -> pd.DataFrame:
    raw = zf.read(xlsx_path)
    # Read the FULL sheet (no usecols) -- the "ELIMINATO" flag can appear in
    # columns past E (e.g. column G for Italy patient 93), so we need every
    # column present to scan for it before narrowing down to the core fields.
    full = pd.read_excel(io.BytesIO(raw), engine="openpyxl", header=0)
    full = full.dropna(how="all")

    # Anything anywhere in the row that says ELIMINATO marks it unusable.
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

    df["excluded"] = df["hgb"].isna() | df["eliminato_flag"]
    df["exclusion_reason"] = ""
    df.loc[df["hgb"].isna(), "exclusion_reason"] = "missing_or_invalid_hgb"
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
# Image extraction
# --------------------------------------------------------------------------
def find_source_files(zf: zipfile.ZipFile, country: str, number: int):
    """Per folder: ignore verified-duplicate '(1)' files, then the sole .jpg is
    the raw image and the sole .png NOT containing 'forniceal' is the palpebral
    crop. This rule is typo-proof (handles 'palplebral', 'papebral', etc.) and
    was verified against all 218 folders with zero exceptions."""
    prefix = f"{ZIP_ROOT}/{country}/{number}/"
    files = [n for n in zf.namelist() if n.startswith(prefix) and n != prefix]
    files = [f for f in files if "(1)" not in f]

    jpgs = [f for f in files if f.lower().endswith(".jpg")]
    crops = [f for f in files if f.lower().endswith(".png") and "forniceal" not in f.lower()]

    if len(jpgs) != 1 or len(crops) != 1:
        raise ValueError(
            f"{country}/{number}: expected 1 jpg + 1 palpebral png, "
            f"found jpgs={jpgs} crops={crops}"
        )
    return jpgs[0], crops[0]


CRITICAL_PNG_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}


def sanitize_png_bytes(data: bytes) -> bytes:
    """Drop ancillary PNG chunks with a corrupted CRC. Verified across this
    dataset: 63/217 palpebral crops have a bad CRC on their 'iCCP' (embedded
    color profile) chunk -- never on IHDR/PLTE/IDAT/IEND, i.e. the actual
    pixel data is always intact. Pillow refuses to open the file at all over
    one bad ancillary chunk, so we strip just that chunk. A bad CRC on a
    critical chunk raises instead, since that would mean real pixel data is
    corrupted rather than optional metadata."""
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


def pad_to_square(img: Image.Image, fill) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new(img.mode, (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def process_patient(zf: zipfile.ZipFile, country: str, number: int, patient_id: str):
    jpg_name, png_name = find_source_files(zf, country, number)

    raw = Image.open(io.BytesIO(zf.read(jpg_name)))
    raw = ImageOps.exif_transpose(raw)  # critical: fixes landscape/portrait mismatch vs. the mask
    raw = raw.convert("RGB")
    raw = pad_to_square(raw, fill=(0, 0, 0))
    raw = raw.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    raw.save(IMAGES_DIR / f"{patient_id}.jpg", quality=95)

    crop = Image.open(io.BytesIO(sanitize_png_bytes(zf.read(png_name))))
    crop_fill = (0, 0, 0, 0) if crop.mode == "RGBA" else (0, 0, 0)
    crop = pad_to_square(crop, fill=crop_fill)
    crop = crop.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    crop.save(MASKS_DIR / f"{patient_id}_palpebral.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        frames = [load_country_metadata(zf, country, path) for country, path in COUNTRIES.items()]
        meta = pd.concat(frames, ignore_index=True)
        meta["anemic_label"] = meta.apply(compute_anemic_label, axis=1)

        included = meta[~meta["excluded"]].copy()
        excluded = meta[meta["excluded"]]
        print(f"Total patients in metadata: {len(meta)}")
        print(f"Excluded: {len(excluded)}")
        if len(excluded):
            print(excluded[["patient_id", "exclusion_reason"]].to_string(index=False))

        for _, row in included.iterrows():
            process_patient(zf, row["country"], row["number"], row["patient_id"])

    included = included.drop(columns=["exclusion_reason", "excluded"])
    included.to_csv(METADATA_CSV, index=False)

    print(f"\nSaved {len(included)} image/mask pairs to {OUTPUT_DIR}")
    print(f"Metadata written to {METADATA_CSV}")

    print("\n--- Class balance (critical: watch for imbalance) ---")
    print(included["anemic_label"].value_counts())
    print("\n--- Anemia rate by country/gender ---")
    print(included.groupby(["country", "gender"])["anemic_label"].mean())


if __name__ == "__main__":
    main()
