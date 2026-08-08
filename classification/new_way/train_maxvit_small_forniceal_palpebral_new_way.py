"""
Entry point (new_way roster): 12-trial Optuna search for MaxViT-Small
(frozen ImageNet backbone, Dropout->Linear head with dropout_rate itself
Optuna-tuned) on the forniceal_palpebral crop, via the shared engine in
../datapreparepipeline/trainer_engine.py. Standalone-runnable (e.g. `python
classification/new_way/train_maxvit_small_forniceal_palpebral_new_way.py`).

MaxViT-Small: Hybrid, medium tier (~80-100M target; verified 68.16M) -- NOT in torchvision, sourced from timm (`timm==1.0.28`, added to requirements.txt 2026-08-08). Pretrained tag `in1k` -- standard ImageNet-1k, same regime as every other model in this project.

Same protocol as every other combo trained through this engine (250-epoch
ceiling, patience=7, dropout_rate tuned {0.2, 0.5}, 12-trial Optuna search) --
only the architecture and tissue type differ.

model_name is "maxvit_small_forniceal_palpebral_new_way" -- the _new_way suffix keeps this roster's
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
        arch_name="maxvit_small",
        tissue_type="forniceal_palpebral",
        model_name="maxvit_small_forniceal_palpebral_new_way",
        n_trials=12,
    )
