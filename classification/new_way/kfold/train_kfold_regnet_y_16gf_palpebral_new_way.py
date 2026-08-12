"""
Entry point: fixed-hyperparameter 3-fold retraining of regnet_y_16gf_palpebral_new_way
(no Optuna -- hyperparameters below are that combo's own best trial from
Output/version1/logs/regnet_y_16gf_palpebral_new_way_study_summary.json, read at generation
time by _generate_scripts.py, not hand-copied).

arch: regnet_y_16gf (CNN) | tissue_type: palpebral
Original single-split result: F1=0.8667 (best trial #8 of 12)

model_name is "regnet_y_16gf_palpebral_new_way_kfold3" -- the _kfold3 suffix keeps this run's
checkpoints/logs from colliding with the version1 Optuna-sweep results
under the un-suffixed name. Standalone-runnable, e.g.
`python classification/new_way/kfold/train_kfold_regnet_y_16gf_palpebral_new_way.py`.
"""

import sys
from pathlib import Path

KFOLD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(KFOLD_DIR))

from kfold_engine import run_kfold_study  # noqa: E402

if __name__ == "__main__":
    run_kfold_study(
        model_name="regnet_y_16gf_palpebral_new_way_kfold3",
        arch_name="regnet_y_16gf",
        tissue_type="palpebral",
        learning_rate=0.006464900601177843,
        weight_decay=1.217404417807552e-06,
        dropout_rate=0.2,
        n_folds=3,
    )
