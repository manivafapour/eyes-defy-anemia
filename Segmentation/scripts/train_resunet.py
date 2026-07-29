"""
Entry point: run the 5-trial Optuna hyperparameter search for Model 3
(ResUNet) via the shared engine in trainer_engine.py. Designed to run
standalone (e.g. `python scripts/train_resunet.py` on Kaggle) with no
shared code to edit -- the model choice lives entirely in this file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.segmentation.resunet import ResUNet  # noqa: E402
from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(model_cls=ResUNet, model_name="resunet")
