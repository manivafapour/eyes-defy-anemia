"""
Entry point (new_way roster): 12-trial Optuna search for ConvNeXt-Large
(frozen ImageNet backbone, Dropout->Linear head with dropout_rate itself
Optuna-tuned) on the palpebral crop, via the shared engine in
../datapreparepipeline/trainer_engine.py. Standalone-runnable (e.g. `python
classification/new_way/train_convnext_large_palpebral_new_way.py`).

ConvNeXt-Large: CNN, heavy tier (120+M target; verified 197.77M constructor)

Same protocol as every other combo trained through this engine (250-epoch
ceiling, patience=7, dropout_rate tuned {0.2, 0.5}, 12-trial Optuna search) --
only the architecture and tissue type differ.

model_name is "convnext_large_palpebral_new_way" -- the _new_way suffix keeps this roster's
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
        arch_name="convnext_large",
        tissue_type="palpebral",
        model_name="convnext_large_palpebral_new_way",
        n_trials=12,
    )
