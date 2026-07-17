"""
Entry point: run the 5-trial Optuna hyperparameter search for Model 1
(Standard U-Net) against the raw-photo-aligned dataset (CLAUDE.md Sec 1.4),
via the shared engine in trainer_engine.py. Designed to run standalone
(e.g. `python scripts/train_standard_unet_aligned.py` on Kaggle) with no
shared code to edit -- the model + dataset choice lives entirely in this file.

model_name is "unet_aligned", not "unet" -- this keeps every output
(outputs/checkpoints/best_unet_aligned.pth, outputs/logs/unet_aligned_*)
distinct from the original crop-based run, so this does not overwrite the
existing best_unet.pth / unet_* logs already produced.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import AlignedConjunctivaSegmentationDataset  # noqa: E402
from models.segmentation.unet import UNet  # noqa: E402
from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(
        model_cls=UNet,
        model_name="unet_aligned",
        dataset_cls=AlignedConjunctivaSegmentationDataset,
    )
