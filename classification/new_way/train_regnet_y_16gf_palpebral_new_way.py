"""
Entry point (new_way roster): 12-trial Optuna search for RegNetY-16GF
(frozen ImageNet backbone, Dropout->Linear head with dropout_rate itself
Optuna-tuned) on the palpebral crop, via the shared engine in
../datapreparepipeline/trainer_engine.py. Standalone-runnable (e.g. `python
classification/new_way/train_regnet_y_16gf_palpebral_new_way.py`).

RegNetY-16GF: CNN, medium tier (~70-80M target; verified 83.59M constructor / 80.57M excl. head)
Uses IMAGENET1K_V1, not a higher-accuracy SWAG variant -- the SWAG variants need 384x384 or 224x224 input, neither matching this project's uniform 256x256 CNN resize (see trainer_engine.py ARCHITECTURE_REGISTRY comment for the full reasoning).

Same protocol as every other combo trained through this engine (250-epoch
ceiling, patience=7, dropout_rate tuned {0.2, 0.5}, 12-trial Optuna search) --
only the architecture and tissue type differ.

model_name is "regnet_y_16gf_palpebral_new_way" -- the _new_way suffix keeps this roster's
checkpoints/logs from colliding with any existing model_name in
classification/outputs/ (03_tech_stack_and_rules.md rule #3).
"""

import sys
from pathlib import Path

DATAPREPAREPIPELINE_DIR = Path(__file__).resolve().parent.parent / "datapreparepipeline"
sys.path.insert(0, str(DATAPREPAREPIPELINE_DIR))

from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(
        arch_name="regnet_y_16gf",
        tissue_type="palpebral",
        model_name="regnet_y_16gf_palpebral_new_way",
        n_trials=12,
    )
