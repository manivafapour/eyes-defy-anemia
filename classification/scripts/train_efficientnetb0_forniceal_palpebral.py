"""
Entry point: 12-trial Optuna search for EfficientNet-B0 (frozen ImageNet
backbone, single-logit head) on the forniceal_palpebral crop, via the
shared engine in trainer_engine.py. Standalone-runnable.

model_name is "efficientnet_b0_forniceal_palpebral" -- keeps
outputs/checkpoints and outputs/logs distinct from the other 5
(architecture, tissue_type) combos. Note: forniceal_palpebral is only
available for 211/217 patients (6 Italy patients have no forniceal
conjunctiva exposed in their source photo, per the dataset's own
documentation) -- TissueClassificationDataset filters to those 211
automatically via extraction_log.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(
        arch_name="efficientnet_b0",
        tissue_type="forniceal_palpebral",
        model_name="efficientnet_b0_forniceal_palpebral",
        n_trials=12,
    )
