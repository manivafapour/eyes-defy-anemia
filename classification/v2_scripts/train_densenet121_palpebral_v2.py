"""
Entry point (v2 protocol): 12-trial Optuna search for DenseNet121 (frozen
ImageNet backbone, Dropout->Linear head with dropout_rate itself
Optuna-tuned) on the palpebral crop, via the shared engine in
../scripts/trainer_engine.py. Standalone-runnable (e.g. `python
classification/v2_scripts/train_densenet121_palpebral_v2.py`).

v2 protocol vs. the original run: MAX_EPOCHS 30->100,
EARLY_STOPPING_PATIENCE 5->7, dropout_rate added as a 3rd Optuna-tuned
hyperparameter (categorical {0.2, 0.5}) for every architecture -- see
classification/.project_memory/02_current_status.md for the full
rationale (the dropout-inconsistency finding in the original 3 models,
the confound-susceptibility hypothesis for transformers, etc).

model_name is "densenet121_palpebral_v2" -- keeps outputs/checkpoints and outputs/logs
distinct from the other 17 (architecture, tissue_type) v2 combos. This
architecture has no v1 baseline (it was not part of the original 6-combo run).
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(
        arch_name="densenet121",
        tissue_type="palpebral",
        model_name="densenet121_palpebral_v2",
        n_trials=12,
    )
