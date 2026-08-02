"""
Entry point (v2 protocol, CLEAN DATA): 12-trial Optuna search for ResNet18
(frozen ImageNet backbone, Dropout->Linear head with dropout_rate itself
Optuna-tuned) on the forniceal_palpebral crop, via the shared engine in
../datapreparepipeline/trainer_engine.py. Standalone-runnable (e.g. `python
classification/v2_clean_scripts/train_resnet18_forniceal_palpebral_v2_clean.py`).

Identical protocol to the original v2 run (100-epoch ceiling, patience=7,
dropout_rate tuned {0.2, 0.5}, 12-trial Optuna search) -- see
classification/.project_memory/02_current_status.md for that rationale.
The ONLY difference from train_resnet18_forniceal_palpebral_v2.py is the data: this run
targets the reprocessed dataset after the white-background bug fix
(2026-08-01, prepare_dataset.py's flatten_to_black() -- see
02_current_status.md "Data bug fixed" entry). The original v2 script is
left completely unmodified.

model_name is "resnet18_forniceal_palpebral_v2_clean" -- the _v2_clean suffix (not a
bare _v2 re-run) is deliberate: it keeps this run's outputs/checkpoints
and outputs/logs from silently overwriting the original v2 run's
dirty-data results under the same filename, so both remain on disk and
comparable side-by-side. Same isolation principle the project already
used for the v1 -> v2 transition (03_tech_stack_and_rules.md rule #3).
"""

import sys
from pathlib import Path

DATAPREPAREPIPELINE_DIR = Path(__file__).resolve().parent.parent / "datapreparepipeline"
sys.path.insert(0, str(DATAPREPAREPIPELINE_DIR))

from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(
        arch_name="resnet18",
        tissue_type="forniceal_palpebral",
        model_name="resnet18_forniceal_palpebral_v2_clean",
        n_trials=12,
    )
