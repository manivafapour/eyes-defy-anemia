"""
Entry point (v2 protocol): 12-trial Optuna search for EfficientNet-B0 (frozen
ImageNet backbone, Dropout->Linear head with dropout_rate itself
Optuna-tuned) on the palpebral crop, via the shared engine in
../datapreparepipeline/trainer_engine.py. Standalone-runnable (e.g. `python
classification/v2_scripts/train_efficientnet_b0_palpebral_v2.py`).

v2 protocol vs. the original run: MAX_EPOCHS 30->100,
EARLY_STOPPING_PATIENCE 5->7, dropout_rate added as a 3rd Optuna-tuned
hyperparameter (categorical {0.2, 0.5}) for every architecture -- see
classification/.project_memory/02_current_status.md for the full
rationale (the dropout-inconsistency finding in the original 3 models,
the confound-susceptibility hypothesis for transformers, etc).

model_name is "efficientnet_b0_palpebral_v2" -- the _v2 suffix keeps outputs/checkpoints and outputs/logs
distinct from the original (pre-dropout, 30-epoch-ceiling) "efficientnet_b0_palpebral"
run, which stays on disk untouched as the superseded v1 baseline.
"""

import sys
from pathlib import Path

DATAPREPAREPIPELINE_DIR = Path(__file__).resolve().parent.parent / "datapreparepipeline"
sys.path.insert(0, str(DATAPREPAREPIPELINE_DIR))

from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(
        arch_name="efficientnet_b0",
        tissue_type="palpebral",
        model_name="efficientnet_b0_palpebral_v2",
        n_trials=12,
    )
