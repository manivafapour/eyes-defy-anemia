"""
Entry point: partial fine-tuning of coatnet_3_palpebral_new_way -- unfreeze
only the attention sub-layer (norm1+attn+ls1) of stages[3].blocks[1] and
continue training from the already-converged frozen-backbone checkpoint
(Output/version1/checkpoints/best_coatnet_3_palpebral_new_way.pth).

Hyperparameters (original learning_rate/weight_decay/dropout_rate) are
read directly from that combo's own real
Output/version1/logs/coatnet_3_palpebral_new_way_study_summary.json, not
hand-copied -- discriminative fine-tune LRs are then derived from the
original learning_rate inside finetune_engine_coatnet3.py.

Standalone-runnable: `python classification/new_way/Fine_tune/train_finetune_coatnet3_palpebral.py`
"""

import json
import sys
from pathlib import Path

FINE_TUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINE_TUNE_DIR))

from finetune_engine_coatnet3 import NEW_WAY_DIR, run_finetune  # noqa: E402

SUMMARY_PATH = NEW_WAY_DIR / "Output" / "version1" / "logs" / "coatnet_3_palpebral_new_way_study_summary.json"

if __name__ == "__main__":
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        best_params = json.load(f)["best_params"]

    run_finetune(
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        dropout_rate=best_params["dropout_rate"],
    )
