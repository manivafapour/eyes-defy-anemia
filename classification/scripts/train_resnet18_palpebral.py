"""
Entry point: 12-trial Optuna search for ResNet18 (frozen ImageNet backbone,
single-logit head) on the palpebral crop, via the shared engine in
trainer_engine.py. Standalone-runnable (e.g. `python
classification/scripts/train_resnet18_palpebral.py`).

model_name is "resnet18_palpebral" -- keeps outputs/checkpoints and
outputs/logs distinct from the other 5 (architecture, tissue_type) combos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(
        arch_name="resnet18",
        tissue_type="palpebral",
        model_name="resnet18_palpebral",
        n_trials=12,
    )
