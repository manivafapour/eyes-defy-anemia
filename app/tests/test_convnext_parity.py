"""Golden parity test: the serving ConvNeXt-Tiny backend reproduces the committed
training result exactly.

The champion checkpoint's validation metrics are frozen in
``classification/v2_clean_scripts/outputs/convnext_tiny_palpebral_v2_clean/
convnext_tiny_palpebral_v2_clean_study_summary.json`` (overall confusion matrix
[[17, 2], [0, 14]], F1 = 0.9333). This test rebuilds the serving backend, runs it
over the same palpebral validation split, and asserts the confusion matrix and F1
match -- proving the reconstructed architecture + eval transform + threshold are
bit-faithful to training (no train/serve skew in Stage 2).

Run:
    python app/tests/test_convnext_parity.py     # standalone
    pytest app/tests/test_convnext_parity.py     # or via pytest

Needs torch, torchvision, albumentations, pandas, plus the local classification data
and the champion checkpoint copied into app/weights/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.models.classification import ConvNeXtTinyClassifier  # noqa: E402

DATA_DIR = REPO_ROOT / "classification" / "data" / "processed"
IMAGES_DIR = DATA_DIR / "images" / "palpebral"
SPLITS_CSV = DATA_DIR / "splits.csv"
EXTRACTION_LOG = DATA_DIR / "extraction_log.csv"
WEIGHTS = REPO_ROOT / "app" / "weights" / "best_convnext_tiny_palpebral_v2_clean.pth"
SUMMARY = (
    REPO_ROOT
    / "classification" / "v2_clean_scripts" / "outputs"
    / "convnext_tiny_palpebral_v2_clean"
    / "convnext_tiny_palpebral_v2_clean_study_summary.json"
)

THRESHOLD = 0.5
TISSUE = "palpebral"


def _val_patients() -> pd.DataFrame:
    splits = pd.read_csv(SPLITS_CSV)
    val = splits[splits["split"] == "val"]
    log = pd.read_csv(EXTRACTION_LOG)
    ok = set(log.loc[log[f"{TISSUE}_status"] == "ok", "patient_id"])
    return val[val["patient_id"].isin(ok)].reset_index(drop=True)


def _confusion(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tp += 1
    return [[tn, fp], [fn, tp]]


def _f1(cm: list[list[int]]) -> float:
    (_tn, fp), (fn, tp) = cm
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def test_convnext_parity() -> None:
    assert WEIGHTS.is_file(), f"Champion checkpoint missing at {WEIGHTS}. Copy it into app/weights/ first."
    clf = ConvNeXtTinyClassifier(weights_path=WEIGHTS, input_size=256, threshold=THRESHOLD, device="cpu")

    df = _val_patients()
    y_true: list[int] = []
    y_pred: list[int] = []
    by_country: dict[str, tuple[list[int], list[int]]] = {"India": ([], []), "Italy": ([], [])}

    for _, row in df.iterrows():
        img = Image.open(IMAGES_DIR / f"{row['patient_id']}.jpg").convert("RGB")
        res = clf.predict(img)
        pred = int(res.probability > THRESHOLD)
        true = int(row["anemic_label"])
        y_true.append(true)
        y_pred.append(pred)
        by_country[row["country"]][0].append(true)
        by_country[row["country"]][1].append(pred)

    cm = _confusion(y_true, y_pred)
    f1 = _f1(cm)

    expected = json.loads(SUMMARY.read_text())["best_val_metrics_by_country"]["overall"]
    exp_cm = expected["confusion_matrix"]
    exp_f1 = expected["f1"]

    print(f"val patients: {len(df)}")
    print(f"overall confusion matrix: got {cm}, expected {exp_cm}")
    print(f"overall F1: got {f1:.4f}, expected {exp_f1:.4f}")
    for country, (yt, yp) in by_country.items():
        print(f"  {country}: {_confusion(yt, yp)} (n={len(yt)})")

    assert cm == exp_cm, f"confusion matrix mismatch: {cm} != {exp_cm}"
    assert abs(f1 - exp_f1) < 1e-3, f"F1 mismatch: {f1} != {exp_f1}"
    print("\nPARITY OK: serving backend reproduces the committed training result exactly.")


if __name__ == "__main__":
    test_convnext_parity()
