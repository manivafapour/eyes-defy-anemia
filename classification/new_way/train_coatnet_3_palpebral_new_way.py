"""
Entry point (new_way roster): 12-trial Optuna search for CoAtNet-3
(frozen ImageNet backbone, Dropout->Linear head with dropout_rate itself
Optuna-tuned) on the palpebral crop, via the shared engine in
../datapreparepipeline/trainer_engine.py. Standalone-runnable (e.g. `python
classification/new_way/train_coatnet_3_palpebral_new_way.py`).

CoAtNet-3: Hybrid, heavy tier (150+M target; verified 163.64M) -- NOT in torchvision, sourced from timm. Pretrained tag `sw_in12k` -- ImageNet-12k, with NO ImageNet-1k fine-tuning stage, unlike every other model in this project (see 03_tech_stack_and_rules.md for the full note).

Same protocol as every other combo trained through this engine (250-epoch
ceiling, patience=7, dropout_rate tuned {0.2, 0.5}, 12-trial Optuna search) --
only the architecture and tissue type differ.

model_name is "coatnet_3_palpebral_new_way" -- the _new_way suffix keeps this roster's
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
        arch_name="coatnet_3",
        tissue_type="palpebral",
        model_name="coatnet_3_palpebral_new_way",
        n_trials=12,
    )
