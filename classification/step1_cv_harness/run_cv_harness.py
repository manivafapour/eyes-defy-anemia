"""
Step 1 -- Measurement Harness: runner.

One combo per invocation by default, so a Kaggle notebook can put each combo
in its own cell and sync results after every one. A 12-combo x 25-fit run is
the longest job this project has attempted; the established pattern here (used
for the v2 batch 1/2 sweeps) is per-cell incremental output rather than a
single monolithic run that loses everything on a timeout.

Usage:
    python classification/step1_cv_harness/run_cv_harness.py --list
    python classification/step1_cv_harness/run_cv_harness.py --combo convnext_tiny_palpebral_v2_clean
    python classification/step1_cv_harness/run_cv_harness.py --all
    python classification/step1_cv_harness/run_cv_harness.py --combo <name> --shuffle-control within_country
    python classification/step1_cv_harness/run_cv_harness.py --combo <name> --repeats 3   # budget mode

Writes, per combo, to outputs/{combo}/:
    oof_predictions.csv   one held-out probability per patient per repeat
    fold_manifest.json    fold assignments + structural check results
    cv_metrics.json       pooled per-repeat AUCs, bootstrap CIs, gate results

No model checkpoints are written -- see cv_config.py, deviation 2.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_config import (  # noqa: E402
    MAX_CI_HALF_WIDTH_INDIA_AUC,
    N_BOOTSTRAP,
    N_REPEATS,
    N_SPLITS,
    OUTPUTS_DIR,
    SEED,
    discover_combos,
)
from cv_data import build_folds, load_held_out, load_pool, pool_composition, shuffle_labels  # noqa: E402
from cv_engine import DEVICE, run_fold  # noqa: E402
from cv_stats import bootstrap_auc_cis  # noqa: E402
from validate_harness import (  # noqa: E402
    check_1_test_isolation,
    check_2_partition_integrity,
    check_3_rare_cell_coverage,
    check_4_pair_counts,
)


def _pooled_per_repeat(pred_df: pd.DataFrame) -> list:
    """Reshape long-format out-of-fold predictions into one patient-aligned
    record per repeat. The shared patient ORDER is enforced here (sort by
    patient_id) because cv_stats' bootstrap resamples a single index vector
    and applies it across repeats -- misaligned ordering would silently pair
    one patient's label with another's probability."""
    per_repeat = []
    for repeat in sorted(pred_df["repeat"].unique()):
        sub = pred_df[pred_df["repeat"] == repeat].sort_values("patient_id").reset_index(drop=True)
        per_repeat.append(
            {
                "repeat": int(repeat),
                "patient_ids": sub["patient_id"].to_numpy(),
                "labels": sub["label"].to_numpy(dtype=float),
                "probs": sub["prob"].to_numpy(dtype=float),
                "countries": sub["country"].to_numpy(),
            }
        )
    return per_repeat


def run_combo(
    combo: dict,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = SEED,
    shuffle_control: str = "none",
    n_bootstrap: int = N_BOOTSTRAP,
    outputs_dir: Path = OUTPUTS_DIR,
) -> dict:
    combo_name = combo["combo_name"]
    tag = combo_name if shuffle_control == "none" else f"{combo_name}__shuffle_{shuffle_control}"
    print(f"\n{'=' * 78}\nStep 1 harness: {tag}\n{'=' * 78}")
    print(f"  device={DEVICE}  arch={combo['arch_name']}  tissue={combo['tissue_type']}")
    print(
        f"  locked hyperparameters (from {combo['source_summary']}): "
        f"lr={combo['learning_rate']:.6g} wd={combo['weight_decay']:.6g} dropout={combo['dropout_rate']}"
    )

    pool_df = load_pool(combo["tissue_type"])
    held_out_df = load_held_out(combo["tissue_type"])
    if shuffle_control != "none":
        print(f"  NEGATIVE CONTROL: labels permuted ({shuffle_control}) -- per-country AUC must collapse to 0.50")
        pool_df = shuffle_labels(pool_df, mode=shuffle_control, seed=seed)

    folds = build_folds(pool_df, n_splits=n_splits, n_repeats=n_repeats, seed=seed)

    # Structural checkpoints re-run here, immediately before training, rather
    # than trusting that validate_harness.py was run earlier -- a run launched
    # with different --folds/--repeats/--seed would otherwise silently skip
    # the geometry that was actually verified.
    print("\n  Structural verification (checkpoints 1-4):")
    checks = [
        check_1_test_isolation(folds, set(held_out_df["patient_id"])),
        check_2_partition_integrity(folds, set(pool_df["patient_id"]), n_splits),
        check_3_rare_cell_coverage(folds, pool_df),
        check_4_pair_counts(pool_df, combo["tissue_type"]),
    ]
    if not all(c["passed"] for c in checks):
        raise SystemExit(f"Structural verification FAILED for {tag}. Refusing to train.")

    combo_dir = outputs_dir / tag
    combo_dir.mkdir(parents=True, exist_ok=True)
    with open(combo_dir / "fold_manifest.json", "w") as f:
        json.dump(
            {
                "combo": combo,
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "seed": seed,
                "shuffle_control": shuffle_control,
                "pool_composition": pool_composition(pool_df),
                "structural_checks": checks,
                "folds": folds,
            },
            f,
            indent=2,
        )

    print(f"\n  Training {len(folds)} fits ({n_repeats} repeats x {n_splits} folds):")
    rows = []
    fold_log = []
    t0 = time.time()
    for i, fold in enumerate(folds, 1):
        result = run_fold(pool_df, fold, combo, base_seed=seed)
        for pid, prob, label, country in zip(
            result["patient_ids"], result["probs"], result["labels"], result["countries"]
        ):
            rows.append(
                {
                    "patient_id": pid,
                    "country": country,
                    "label": float(label),
                    "prob": float(prob),
                    "repeat": result["repeat"],
                    "fold": result["fold"],
                }
            )
        fold_log.append(
            {k: result[k] for k in ("repeat", "fold", "n_epochs_run", "best_inner_epoch", "best_inner_loss")}
        )
        if i % n_splits == 0:
            print(f"    -- repeat {result['repeat']} complete ({i}/{len(folds)} fits, "
                  f"{time.time() - t0:.0f}s elapsed)", flush=True)

    pred_df = pd.DataFrame(rows)
    pred_path = combo_dir / "oof_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    per_repeat = _pooled_per_repeat(pred_df)
    stats = bootstrap_auc_cis(per_repeat, n_bootstrap=n_bootstrap, seed=seed)

    india = stats["India"]
    half_width = india.get("ci_half_width")
    precision_gate = half_width is not None and half_width <= MAX_CI_HALF_WIDTH_INDIA_AUC

    metrics = {
        "combo": combo,
        "tag": tag,
        "shuffle_control": shuffle_control,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "pool_composition": pool_composition(pool_df),
        "auc": stats,
        "fold_log": fold_log,
        "runtime_seconds": round(time.time() - t0, 1),
        "gates": {
            "precision_india_ci_half_width": {
                "value": half_width,
                "threshold": MAX_CI_HALF_WIDTH_INDIA_AUC,
                "passed": bool(precision_gate),
            }
        },
    }
    with open(combo_dir / "cv_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  --- {tag} ---")
    for key in ("overall", "India", "Italy", "india_italy_gap"):
        s = stats[key]
        if s["point"] is None:
            print(f"    {key:<18} undefined")
            continue
        line = (
            f"    {key:<18} {s['point']:.4f}  95% CI [{s['ci_low']:.4f}, {s['ci_high']:.4f}]"
            f"  (+/-{s['ci_half_width']:.4f})"
        )
        if s["sd_across_repeats"] is not None:
            line += f"  SD across repeats={s['sd_across_repeats']:.4f}"
        print(line)
    print(f"    precision gate (India CI half-width <= {MAX_CI_HALF_WIDTH_INDIA_AUC}): "
          f"{'PASS' if precision_gate else 'FAIL'}")
    print(f"    saved -> {pred_path.parent}")
    return metrics


def main() -> int:
    combos = discover_combos()
    parser = argparse.ArgumentParser(description="Step 1 measurement harness (pooled out-of-fold repeated CV).")
    parser.add_argument("--combo", help="Combo name, e.g. convnext_tiny_palpebral_v2_clean")
    parser.add_argument("--all", action="store_true", help="Run every discovered combo sequentially.")
    parser.add_argument("--list", action="store_true", help="List discovered combos and locked hyperparameters.")
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS, help="Budget fallback: 3.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument(
        "--shuffle-control",
        choices=["none", "within_country", "global"],
        default="none",
        help="Checkpoint 5. 'within_country' preserves each country's label rate while destroying "
             "per-patient signal: per-country AUC must fall to 0.50.",
    )
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    if args.list:
        print(f"{len(combos)} combos discovered (source: each combo's own v2_clean study summary):\n")
        for name, c in combos.items():
            print(f"  {name}")
            print(f"      arch={c['arch_name']:<20} tissue={c['tissue_type']:<20} "
                  f"lr={c['learning_rate']:.6g} wd={c['weight_decay']:.6g} dropout={c['dropout_rate']}")
        return 0

    if args.all:
        targets = list(combos.values())
    elif args.combo:
        if args.combo not in combos:
            print(f"Unknown combo {args.combo!r}. Available:\n  " + "\n  ".join(combos))
            return 1
        targets = [combos[args.combo]]
    else:
        parser.error("one of --combo, --all or --list is required")

    for combo in targets:
        run_combo(
            combo,
            n_splits=args.folds,
            n_repeats=args.repeats,
            seed=args.seed,
            shuffle_control=args.shuffle_control,
            n_bootstrap=args.bootstrap,
            outputs_dir=args.outputs_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
