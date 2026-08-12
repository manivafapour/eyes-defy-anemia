"""
Generates one train_kfold_{model_name}.py entry-point script per each of
the 6 best new_way combos (by validation F1, from
Output/version1/compare/new_way_model_comparison.xlsx), for the
fixed-hyperparameter 3-fold retraining in kfold_engine.py.

The 6 target combos are a one-time human ranking decision (top 6 by F1 in
the comparison Excel -- ranks 1,1,3,3,5,6, two exact ties) and are named
explicitly below, not re-derived by this script -- but every hyperparameter
(learning_rate/weight_decay/dropout_rate) and the architecture/tissue_type
are read directly from each combo's own real
Output/version1/logs/{model_name}_study_summary.json, never hand-copied,
so there's no transcription step to get wrong (same anti-transcription-
error precedent as every other *_generate_scripts.py in this project).

Run: python _generate_scripts.py
"""

import json
from pathlib import Path

KFOLD_DIR = Path(__file__).resolve().parent
NEW_WAY_DIR = KFOLD_DIR.parent
VERSION1_LOGS_DIR = NEW_WAY_DIR / "Output" / "version1" / "logs"

# Top 6 by validation F1 in new_way_model_comparison.xlsx.
TARGET_MODELS = [
    "convnext_base_palpebral_new_way",
    "convnext_large_palpebral_new_way",
    "coatnet_3_palpebral_new_way",
    "efficientnet_b3_forniceal_palpebral_new_way",
    "maxvit_t_palpebral_new_way",
    "regnet_y_16gf_palpebral_new_way",
]

TISSUE_TYPES = ["palpebral", "forniceal_palpebral"]
CNN_ARCHS = {"efficientnet_b3", "efficientnet_b4", "regnet_y_16gf", "convnext_base", "convnext_large"}


def parse_arch_and_tissue(model_name: str) -> tuple:
    stem = model_name[: -len("_new_way")] if model_name.endswith("_new_way") else model_name
    for tissue in sorted(TISSUE_TYPES, key=len, reverse=True):  # forniceal_palpebral checked before palpebral
        if stem.endswith(f"_{tissue}"):
            return stem[: -len(f"_{tissue}")], tissue
    raise ValueError(f"Could not parse tissue type from model_name={model_name!r}")


TEMPLATE = '''"""
Entry point: fixed-hyperparameter 3-fold retraining of {model_name}
(no Optuna -- hyperparameters below are that combo's own best trial from
Output/version1/logs/{model_name}_study_summary.json, read at generation
time by _generate_scripts.py, not hand-copied).

arch: {arch_name} ({family}) | tissue_type: {tissue_type}
Original single-split result: F1={orig_f1:.4f} (best trial #{orig_trial} of {orig_n_trials})

model_name is "{kfold_model_name}" -- the _kfold3 suffix keeps this run's
checkpoints/logs from colliding with the version1 Optuna-sweep results
under the un-suffixed name. Standalone-runnable, e.g.
`python classification/new_way/kfold/train_kfold_{model_name}.py`.
"""

import sys
from pathlib import Path

KFOLD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(KFOLD_DIR))

from kfold_engine import run_kfold_study  # noqa: E402

if __name__ == "__main__":
    run_kfold_study(
        model_name="{kfold_model_name}",
        arch_name="{arch_name}",
        tissue_type="{tissue_type}",
        learning_rate={learning_rate!r},
        weight_decay={weight_decay!r},
        dropout_rate={dropout_rate!r},
        n_folds=3,
    )
'''


def main():
    generated = []
    for model_name in TARGET_MODELS:
        summary_path = VERSION1_LOGS_DIR / f"{model_name}_study_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"{summary_path} not found -- is TARGET_MODELS out of date?")
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        arch_name, tissue_type = parse_arch_and_tissue(model_name)
        params = summary["best_params"]
        kfold_model_name = f"{model_name}_kfold3"

        script = TEMPLATE.format(
            model_name=model_name,
            arch_name=arch_name,
            tissue_type=tissue_type,
            family="CNN" if arch_name in CNN_ARCHS else "Hybrid",
            orig_f1=summary["best_val_f1"],
            orig_trial=summary["best_trial_number"],
            orig_n_trials=summary["n_trials_run"],
            kfold_model_name=kfold_model_name,
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            dropout_rate=params["dropout_rate"],
        )

        out_path = KFOLD_DIR / f"train_kfold_{model_name}.py"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(script)
        generated.append(out_path.name)
        print(
            f"Generated {out_path.name}  (arch={arch_name} tissue={tissue_type} "
            f"lr={params['learning_rate']:.4g} wd={params['weight_decay']:.4g} dropout={params['dropout_rate']})"
        )

    print(f"\n{len(generated)} entry-point scripts written to {KFOLD_DIR}")


if __name__ == "__main__":
    main()
