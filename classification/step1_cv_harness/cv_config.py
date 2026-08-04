"""
Step 1 -- Measurement Harness: configuration.

PURPOSE OF THIS MODULE (read before changing anything here):
Step 1 does NOT try to improve any model. It replaces a statistically
unusable estimator with a usable one, and produces the frozen baseline that
Steps 2-5 (diagnostics, loss/sampling interventions, image-space
interventions, partial unfreezing) are measured against.

The problem being fixed: the single 70/15/15 split gives a validation set of
33 patients -- 14 India (10 anemic / 4 healthy) and 19 Italy (4 anemic / 15
healthy). India AUC is therefore computed over 10 x 4 = 40 discordant pairs,
so it can only take values in multiples of 1/40 (which is exactly what the
v2_clean results show -- every India AUC in that table is a multiple of
0.025) and carries a 95% CI half-width of roughly +/- 0.27. Under that noise
floor, the observed spread of India AUC across the 12 CNN combos
(0.550-1.000) is not a ranking of models, and no intervention in Steps 3-5
could be shown to work.

The fix: pooled out-of-fold repeated stratified cross-validation over the
train+val pool, which raises the India pair count from 40 to 1,311 (57
anemic x 23 healthy) and the Italy pair count from 60 to 1,680 (20 x 84).

DELIBERATE DEVIATIONS FROM THE v2_clean PROTOCOL (both are corrections, and
both are why this is a fresh module rather than a re-run):

1. Early stopping runs on an INNER validation split carved out of each
   fold's own training portion, never on the outer held-out fold. The
   v2_clean protocol early-stopped on the same split it then reported, which
   selects the reported epoch using the evaluation data. That is a real (if
   small) optimistic bias, and Step 1's entire purpose is a defensible
   number, so it is removed here rather than reproduced.
2. No per-fold checkpoints are written. Step 1 needs out-of-fold
   PREDICTIONS, not weights; writing 25 checkpoints x 12 combos would be
   tens of GB of Kaggle output for artifacts nothing downstream reads.

Everything else -- architecture builders, frozen backbone, head shape,
transforms, augmentation, loss, optimizer, epoch ceiling, patience -- is
imported from or mirrored exactly against the live pipeline
(classification/datapreparepipeline/), so this measures the SAME models the
v2_clean sweep produced. Any change to those would make Step 1 something
other than a baseline.
"""

import json
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = HARNESS_DIR.parent           # classification/
PIPELINE_DIR = MODULE_ROOT / "datapreparepipeline"

# The live pipeline is imported, NOT copied -- Step 1 must build bit-identical
# models to the ones v2_clean trained, and a forked copy of the builders would
# silently drift away from that guarantee the first time either side is edited.
# (The deprecated datapreparepipeline/efficientnet_b0_forniceal_5fold_cv/ is
# deliberately not referenced anywhere in this module.)
sys.path.insert(0, str(PIPELINE_DIR))

PROCESSED_DIR = MODULE_ROOT / "data" / "processed"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"
EXTRACTION_LOG_CSV = PROCESSED_DIR / "extraction_log.csv"
IMAGES_DIR = PROCESSED_DIR / "images"

# Source of the locked hyperparameters. Each combo reuses its own winning
# Optuna trial verbatim -- re-tuning inside the new protocol would confound
# "better measurement" with "better hyperparameters", and Step 1 must isolate
# the former.
V2_CLEAN_OUTPUTS_DIR = MODULE_ROOT / "v2_clean_scripts" / "outputs"

OUTPUTS_DIR = HARNESS_DIR / "outputs"
BASELINE_DIR = OUTPUTS_DIR / "baseline"

# --------------------------------------------------------------------------
# Cross-validation design
# --------------------------------------------------------------------------
# Pool = train + val. The 33-patient test split is removed once, here, and is
# asserted absent from every fold by validate_harness.py check 1 -- it stays
# sealed for Step 6 and must not be touched by any of Steps 1-5.
POOL_SPLITS = ("train", "val")
HELD_OUT_SPLIT = "test"

N_SPLITS = 5
N_REPEATS = 5          # 25 fits per combo; --repeats 3 is the budget fallback
SEED = 42

# Stratification is on the compound country x label key, NOT the label alone.
# This is the single most important line in the file: India-healthy is only 23
# patients in the pool, and label-only stratification can hand an individual
# fold 2 of them, which would make that fold's India AUC meaningless and
# corrupt the pooled estimate.
STRATUM_COL = "stratum"

# Inner split used only for early stopping (see deviation 1 above). 15% of each
# fold's training portion, itself stratified on the same 4-cell key so the
# rarest cells stay represented on both sides of the inner cut.
INNER_VAL_FRACTION = 0.15

# --------------------------------------------------------------------------
# Training protocol -- mirrors datapreparepipeline/trainer_engine.py's v2
# constants. Defined locally rather than imported so that a future edit to the
# live engine cannot silently redefine what this frozen baseline meant.
# Verified equal to the live values at the time of writing.
# --------------------------------------------------------------------------
MAX_EPOCHS = 250
EARLY_STOPPING_PATIENCE = 7
BATCH_SIZE = 32
NUM_WORKERS = 0        # Kaggle-safe; matches the live pipeline

# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
N_BOOTSTRAP = 2000
CI_ALPHA = 0.05

# --------------------------------------------------------------------------
# Gates (checkpoint 8). These are pass/fail criteria for clearing Step 1, not
# targets to optimize.
# --------------------------------------------------------------------------
MAX_CI_HALF_WIDTH_INDIA_AUC = 0.12
MIN_RARE_CELL_PER_FOLD = 3          # >=3 India-healthy and >=3 Italy-anemic per fold
PLAUSIBILITY_RANGE_INDIA_AUC = (0.40, 0.90)   # outside this => suspect the harness, not the model

COUNTRIES = ["India", "Italy"]


# --------------------------------------------------------------------------
# Combo discovery + locked hyperparameters
# --------------------------------------------------------------------------
def _parse_combo_name(model_name: str) -> tuple:
    """'convnext_tiny_forniceal_palpebral_v2_clean' -> ('convnext_tiny',
    'forniceal_palpebral'). Parsed against the real architecture registry
    rather than split on underscores -- both architecture names and tissue
    names contain underscores, so a positional split would silently
    mis-assign (e.g. 'mobilenet_v3_small' + 'palpebral')."""
    from trainer_engine import ARCHITECTURE_REGISTRY  # noqa: PLC0415

    stem = model_name[: -len("_v2_clean")] if model_name.endswith("_v2_clean") else model_name
    for arch in sorted(ARCHITECTURE_REGISTRY, key=len, reverse=True):
        if stem.startswith(arch + "_"):
            tissue = stem[len(arch) + 1 :]
            if tissue in ("palpebral", "forniceal_palpebral"):
                return arch, tissue
    raise ValueError(f"Could not parse architecture/tissue from model_name {model_name!r}")


def discover_combos() -> dict:
    """Discovers every combo that has a completed v2_clean study, and reads
    its winning hyperparameters straight out of that study's own summary
    JSON. Dynamic discovery, not a hardcoded list, for two reasons: the 6
    transformer combos have not been run yet and will join automatically once
    their sweep lands, and hand-transcribing 12 sets of float hyperparameters
    is exactly the kind of silent-error surface this project has been bitten
    by before.

    Returns {combo_name: {arch_name, tissue_type, learning_rate,
    weight_decay, dropout_rate, source_summary}}.
    """
    combos = {}
    for summary_path in sorted(V2_CLEAN_OUTPUTS_DIR.glob("*/*_study_summary.json")):
        with open(summary_path) as f:
            summary = json.load(f)
        model_name = summary["model_name"]
        arch_name, tissue_type = _parse_combo_name(model_name)
        params = summary["best_params"]
        combos[model_name] = {
            "combo_name": model_name,
            "arch_name": arch_name,
            "tissue_type": tissue_type,
            "learning_rate": float(params["learning_rate"]),
            "weight_decay": float(params["weight_decay"]),
            "dropout_rate": float(params["dropout_rate"]),
            "source_summary": str(summary_path.relative_to(MODULE_ROOT)),
            "source_best_val_f1": summary.get("best_val_f1"),
        }
    if not combos:
        raise FileNotFoundError(
            f"No *_study_summary.json found under {V2_CLEAN_OUTPUTS_DIR}. "
            "Step 1 reuses each combo's winning Optuna hyperparameters, so the "
            "v2_clean results must be present (they are git-tracked; on Kaggle "
            "they arrive with the repo clone)."
        )
    return combos


def fit_seed(base_seed: int, repeat: int, fold: int) -> int:
    """One deterministic seed per (repeat, fold) fit, so a re-run of any
    single fold reproduces exactly (checkpoint 6) without having to replay
    every fit before it."""
    return base_seed * 1000 + repeat * 100 + fold
