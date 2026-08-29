"""
Entry point: bagging and boosting over the frozen-backbone head, for a
small subset of architectures (unlike ensemble_engine.py's techniques,
these train genuinely NEW models -- see bagging_boosting.py's module
docstring for why that limits scope here).

max_epochs=60/patience=8 for bagging mirrors kfold_engine.py's own budget
loosely (250/15) scaled down, since these OOB-validated rounds converge on
a similar timescale to the original 3 folds (which mostly early-stopped in
the 20-50 epoch range per Output/version2/logs/*_kfold_folds.csv).
epochs_per_round=25 (fixed, no early stopping -- no held-out set exists
mid-boosting) for boosting is a middle-of-the-road guess at the same
convergence scale; a smoke test at 5 epochs/round clearly underfit
(round-1 own test_f1=0.35), so this is deliberately higher, not arbitrary.

Standalone-runnable: `python classification/new_way/kfold/ensemble/run_bagging_boosting.py`
"""

import json
import sys
from pathlib import Path

ENSEMBLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENSEMBLE_DIR))

from bagging_boosting import BB_CACHE_DIR, run_bagging, run_boosting  # noqa: E402

if __name__ == "__main__":
    model_keys = ["regnet_y_16gf", "maxvit_t"]
    results = {"bagging": {}, "boosting": {}}

    for model_key in model_keys:
        print(f"\n{'=' * 70}\nBAGGING: {model_key}\n{'=' * 70}")
        results["bagging"][model_key] = run_bagging(model_key, n_bags=5, max_epochs=60, patience=8)

        print(f"\n{'=' * 70}\nBOOSTING: {model_key}\n{'=' * 70}")
        results["boosting"][model_key] = run_boosting(model_key, n_rounds=5, epochs_per_round=25)

    BB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = BB_CACHE_DIR / "bagging_boosting_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for technique in ["bagging", "boosting"]:
        for model_key, r in results[technique].items():
            m = r["final_metrics"]["overall"]
            print(f"  {technique:10s} {model_key:16s} f1={m['f1']:.4f}  acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")
    print(f"\nSaved to {summary_path}")
