"""
Entry point: partial fine-tuning of efficientnet_b3_forniceal_palpebral_new_way
-- unfreeze the full last MBConv block (features[7][1], including its 3
BatchNorm layers) and continue training from the already-converged
frozen-backbone checkpoint (Output/version1/checkpoints/
best_efficientnet_b3_forniceal_palpebral_new_way.pth).

Hyperparameters (original learning_rate/weight_decay/dropout_rate) are
read directly from that combo's own real
Output/version1/logs/efficientnet_b3_forniceal_palpebral_new_way_study_summary.json,
not hand-copied.

Standalone-runnable: `python classification/new_way/Fine_tune/train_finetune_efficientnet_b3_forniceal.py`
"""

import json
import sys
from pathlib import Path

FINE_TUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINE_TUNE_DIR))

from finetune_engine_efficientnet_b3 import NEW_WAY_DIR, run_finetune  # noqa: E402

SUMMARY_PATH = NEW_WAY_DIR / "Output" / "version1" / "logs" / "efficientnet_b3_forniceal_palpebral_new_way_study_summary.json"

if __name__ == "__main__":
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        best_params = json.load(f)["best_params"]

    run_finetune(
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        dropout_rate=best_params["dropout_rate"],
    )
