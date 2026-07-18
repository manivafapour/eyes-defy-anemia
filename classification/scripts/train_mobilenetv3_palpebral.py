"""
Entry point: 12-trial Optuna search for MobileNetV3-Small (frozen ImageNet
backbone, single-logit head) on the palpebral crop, via the shared engine
in trainer_engine.py. Standalone-runnable.

model_name is "mobilenet_v3_small_palpebral" -- keeps outputs/checkpoints
and outputs/logs distinct from the other 5 (architecture, tissue_type)
combos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(
        arch_name="mobilenet_v3_small",
        tissue_type="palpebral",
        model_name="mobilenet_v3_small_palpebral",
        n_trials=12,
    )
