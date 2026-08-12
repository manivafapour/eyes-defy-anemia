"""
Entry point: fixed-hyperparameter 3-fold retraining of coatnet_3_palpebral_new_way
(no Optuna -- hyperparameters below are that combo's own best trial from
Output/version1/logs/coatnet_3_palpebral_new_way_study_summary.json, read at generation
time by _generate_scripts.py, not hand-copied).

arch: coatnet_3 (Hybrid) | tissue_type: palpebral
Original single-split result: F1=0.8966 (best trial #9 of 12)

model_name is "coatnet_3_palpebral_new_way_kfold3" -- the _kfold3 suffix keeps this run's
checkpoints/logs from colliding with the version1 Optuna-sweep results
under the un-suffixed name. Standalone-runnable, e.g.
`python classification/new_way/kfold/train_kfold_coatnet_3_palpebral_new_way.py`.
"""

import sys
from pathlib import Path

KFOLD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(KFOLD_DIR))

from kfold_engine import run_kfold_study  # noqa: E402

if __name__ == "__main__":
    run_kfold_study(
        model_name="coatnet_3_palpebral_new_way_kfold3",
        arch_name="coatnet_3",
        tissue_type="palpebral",
        learning_rate=0.0011249436130085291,
        weight_decay=0.00011856737112037188,
        dropout_rate=0.2,
        n_folds=3,
    )
