"""
Entry point: run the 12-trial Optuna hyperparameter search for Model 2
(Attention U-Net) against the raw-photo-aligned dataset (CLAUDE.md Sec 1.4),
via the shared engine in trainer_engine.py. Designed to run standalone
(e.g. `python scripts/train_attention_unet_aligned.py` on Kaggle) with no
shared code to edit -- the model + dataset choice lives entirely in this file.

model_name is "attention_unet_aligned", not "attention_unet" -- this keeps
every output distinct from the original crop-based run.

n_trials=12 (not the engine's 5-trial default, CLAUDE.md Sec 3.4) --
loss_fn is now a 3rd Optuna-tuned dimension alongside learning_rate/
weight_decay (Sec 3.2b), and 5 trials gives too little coverage to compare
bce_dice vs. focal_tversky meaningfully.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import AlignedConjunctivaSegmentationDataset  # noqa: E402
from models.segmentation.attention_unet import AttentionUNet  # noqa: E402
from trainer_engine import run_study  # noqa: E402

if __name__ == "__main__":
    run_study(
        model_cls=AttentionUNet,
        model_name="attention_unet_aligned",
        dataset_cls=AlignedConjunctivaSegmentationDataset,
        n_trials=12,
    )
