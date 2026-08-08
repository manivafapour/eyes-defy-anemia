"""
Step 1 -- Measurement Harness: baseline aggregation and gate evaluation
(checkpoints 5, 7, 8, 9).

Reads every completed combo's cv_metrics.json and produces the frozen Step 1
baseline artifact. This file is the reference every later step compares
against, so it records not just the numbers but the exact configuration that
produced them (seed, folds, repeats, bootstrap count, and each combo's locked
hyperparameters) -- a baseline whose provenance is not recorded cannot support
a paired comparison later.

Fails loudly and refuses to declare Step 1 clear if any gate fails.

Also writes outputs/comparison/ -- cross-combo plots (pooled India/Italy/
overall AUC with bootstrap CI whiskers, and the India/Italy gap), separate
from outputs/baseline/'s numeric artifact so the two "what changed" concerns
(numbers vs. visualization) stay in their own files.

Usage:
    python classification/step1_cv_harness/aggregate_baseline.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_config import (  # noqa: E402
    BASELINE_DIR,
    COMPARISON_DIR,
    MAX_CI_HALF_WIDTH_INDIA_AUC,
    OUTPUTS_DIR,
    PLAUSIBILITY_RANGE_INDIA_AUC,
)
from cv_plots import plot_country_auc_comparison, plot_gap_comparison  # noqa: E402

SINGLE_SPLIT_PAIRS = {"India": 40, "Italy": 60}


def _load_metrics(outputs_dir: Path) -> tuple:
    real, controls = [], []
    for path in sorted(outputs_dir.glob("*/cv_metrics.json")):
        with open(path) as f:
            m = json.load(f)
        (controls if m.get("shuffle_control", "none") != "none" else real).append(m)
    return real, controls


def _row(m: dict) -> dict:
    auc = m["auc"]

    def g(key, field):
        return auc[key][field]

    return {
        "combo": m["combo"]["combo_name"],
        "arch": m["combo"]["arch_name"],
        "tissue": m["combo"]["tissue_type"],
        "india_auc": g("India", "point"),
        "india_ci_low": g("India", "ci_low"),
        "india_ci_high": g("India", "ci_high"),
        "india_ci_half_width": g("India", "ci_half_width"),
        "india_sd_across_repeats": g("India", "sd_across_repeats"),
        "italy_auc": g("Italy", "point"),
        "italy_ci_low": g("Italy", "ci_low"),
        "italy_ci_high": g("Italy", "ci_high"),
        "overall_auc": g("overall", "point"),
        "overall_ci_low": g("overall", "ci_low"),
        "overall_ci_high": g("overall", "ci_high"),
        "gap": g("india_italy_gap", "point"),
        "gap_ci_low": g("india_italy_gap", "ci_low"),
        "gap_ci_high": g("india_italy_gap", "ci_high"),
        "gap_ci_excludes_zero": (
            None
            if g("india_italy_gap", "ci_low") is None
            else bool(g("india_italy_gap", "ci_low") > 0 or g("india_italy_gap", "ci_high") < 0)
        ),
        "india_pairs": m["pool_composition"]["India"]["n_discordant_pairs"],
        "italy_pairs": m["pool_composition"]["Italy"]["n_discordant_pairs"],
        "n_repeats": m["n_repeats"],
        "n_splits": m["n_splits"],
        "seed": m["seed"],
        "learning_rate": m["combo"]["learning_rate"],
        "weight_decay": m["combo"]["weight_decay"],
        "dropout_rate": m["combo"]["dropout_rate"],
        "runtime_seconds": m.get("runtime_seconds"),
    }


def evaluate_gates(df: pd.DataFrame, controls: list) -> dict:
    """Checkpoints 5, 7, 8. Checkpoint 9 is the artifact this script writes."""
    gates = {}

    # --- Checkpoint 8: precision. The pass/fail criterion for Step 1. If this
    # fails, Steps 3-5 cannot demonstrate anything and their success criteria
    # have to be renegotiated rather than quietly ignored.
    failures = df[df["india_ci_half_width"] > MAX_CI_HALF_WIDTH_INDIA_AUC]
    gates["8_precision"] = {
        "threshold_ci_half_width": MAX_CI_HALF_WIDTH_INDIA_AUC,
        "worst_observed": float(df["india_ci_half_width"].max()) if len(df) else None,
        "median_observed": float(df["india_ci_half_width"].median()) if len(df) else None,
        "n_failing": int(len(failures)),
        "failing_combos": failures["combo"].tolist(),
        "passed": bool(len(failures) == 0),
    }

    # --- Checkpoint 7: plausibility. A pooled India AUC far outside the range
    # the single-split CIs already permitted is evidence of a harness bug, not
    # a discovery -- investigate before reporting.
    lo, hi = PLAUSIBILITY_RANGE_INDIA_AUC
    implausible = df[(df["india_auc"] < lo) | (df["india_auc"] > hi)]
    gates["7_plausibility"] = {
        "expected_range": [lo, hi],
        "observed_min": float(df["india_auc"].min()) if len(df) else None,
        "observed_max": float(df["india_auc"].max()) if len(df) else None,
        "n_outside": int(len(implausible)),
        "combos_outside": implausible["combo"].tolist(),
        "passed": bool(len(implausible) == 0),
        "note": "Outside this range => suspect the harness before believing the result.",
    }

    # --- Checkpoint 5: label-shuffle negative control.
    control_results = []
    for m in controls:
        auc = m["auc"]
        india_covers = (
            auc["India"]["ci_low"] is not None
            and auc["India"]["ci_low"] <= 0.5 <= auc["India"]["ci_high"]
        )
        italy_covers = (
            auc["Italy"]["ci_low"] is not None
            and auc["Italy"]["ci_low"] <= 0.5 <= auc["Italy"]["ci_high"]
        )
        mode = m["shuffle_control"]
        overall_covers = (
            auc["overall"]["ci_low"] is not None
            and auc["overall"]["ci_low"] <= 0.5 <= auc["overall"]["ci_high"]
        )
        entry = {
            "combo": m["combo"]["combo_name"],
            "mode": mode,
            "india_auc": auc["India"]["point"],
            "india_ci": [auc["India"]["ci_low"], auc["India"]["ci_high"]],
            "italy_auc": auc["Italy"]["point"],
            "italy_ci": [auc["Italy"]["ci_low"], auc["Italy"]["ci_high"]],
            "overall_auc": auc["overall"]["point"],
            "overall_ci": [auc["overall"]["ci_low"], auc["overall"]["ci_high"]],
            # Leakage test: with labels destroyed, no per-country AUC may be
            # distinguishable from chance.
            "per_country_at_chance": bool(india_covers and italy_covers),
        }
        if mode == "global":
            entry["overall_at_chance"] = bool(overall_covers)
            entry["passed"] = bool(india_covers and italy_covers and overall_covers)
        else:
            # within_country deliberately preserves each country's label rate,
            # so overall AUC is EXPECTED to sit above chance even though the
            # model has learned nothing about any individual patient. That is
            # not a failure -- it is a direct demonstration of the
            # country-shortcut mechanism, and is recorded as such.
            entry["overall_above_chance_as_expected"] = bool(not overall_covers)
            entry["passed"] = bool(india_covers and italy_covers)
        control_results.append(entry)

    gates["5_negative_control"] = {
        "n_controls_run": len(control_results),
        "results": control_results,
        "passed": bool(control_results) and all(c["passed"] for c in control_results),
        "note": (
            "Not run -- Step 1 cannot be cleared without at least one label-shuffle control."
            if not control_results
            else "Per-country AUC must cover 0.50 once labels are destroyed; otherwise the harness leaks."
        ),
    }
    return gates


def write_markdown(df: pd.DataFrame, gates: dict, meta: dict, path: Path) -> None:
    lines = [
        "# Step 1 Baseline -- Pooled Out-of-Fold Repeated Cross-Validation",
        "",
        f"Generated: {meta['timestamp_utc']}",
        "",
        "Frozen reference for Steps 2-5. Every later intervention must be compared against this",
        "using a **paired** bootstrap on the difference (`cv_stats.paired_delta_auc`) with the",
        "identical fold assignments recorded in each combo's `fold_manifest.json` -- not by",
        "checking whether two independent confidence intervals overlap.",
        "",
        "## Configuration",
        "",
        f"- Design: {meta['n_splits']}-fold x {meta['n_repeats']} repeats, stratified on country x label",
        f"- Seed: {meta['seed']}",
        f"- Bootstrap replicates: {meta['n_bootstrap']}",
        "- Pool: train + val only; the 33-patient test split is sealed for Step 6",
        "- Hyperparameters: each combo's own winning Optuna trial, reused verbatim (no re-tuning)",
        "",
        "## Precision achieved",
        "",
        f"| | India pairs | Italy pairs |",
        f"|---|---|---|",
        f"| Single 70/15/15 split | {SINGLE_SPLIT_PAIRS['India']} | {SINGLE_SPLIT_PAIRS['Italy']} |",
        f"| Pooled out-of-fold | {int(df['india_pairs'].max())} | {int(df['italy_pairs'].max())} |",
        "",
        "## Results (sorted by India AUC)",
        "",
        "| Combo | India AUC [95% CI] | Italy AUC [95% CI] | Overall AUC | Gap [95% CI] | Gap excl. 0 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.sort_values("india_auc", ascending=False).iterrows():
        lines.append(
            f"| {r['combo'].replace('_v2_clean', '')} "
            f"| {r['india_auc']:.3f} [{r['india_ci_low']:.3f}, {r['india_ci_high']:.3f}] "
            f"| {r['italy_auc']:.3f} [{r['italy_ci_low']:.3f}, {r['italy_ci_high']:.3f}] "
            f"| {r['overall_auc']:.3f} "
            f"| {r['gap']:+.3f} [{r['gap_ci_low']:+.3f}, {r['gap_ci_high']:+.3f}] "
            f"| {'yes' if r['gap_ci_excludes_zero'] else 'no'} |"
        )

    lines += ["", "## Gate results", ""]
    for name, gate in sorted(gates.items()):
        lines.append(f"### {name} -- {'PASS' if gate['passed'] else 'FAIL'}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(gate, indent=2))
        lines.append("```")
        lines.append("")

    all_passed = all(g["passed"] for g in gates.values())
    lines += [
        "## Verdict",
        "",
        (
            "**Step 1 CLEAR.** The measurement harness is verified and the baseline is frozen. "
            "Step 2 (diagnostics) may proceed."
            if all_passed
            else "**Step 1 NOT CLEAR.** One or more gates failed -- resolve before starting Step 2. "
            "Running interventions against an unverified harness would produce unfalsifiable results."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Step 1 results and evaluate gates 5, 7, 8, 9.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--comparison-dir", type=Path, default=COMPARISON_DIR)
    args = parser.parse_args()

    real, controls = _load_metrics(args.outputs_dir)
    if not real:
        print(f"No completed combos found under {args.outputs_dir}. Run run_cv_harness.py first.")
        return 1

    df = pd.DataFrame([_row(m) for m in real])
    gates = evaluate_gates(df, controls)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_combos": len(df),
        "n_splits": int(real[0]["n_splits"]),
        "n_repeats": int(real[0]["n_repeats"]),
        "seed": int(real[0]["seed"]),
        "n_bootstrap": int(real[0]["n_bootstrap"]),
    }

    args.baseline_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.baseline_dir / "step1_baseline.csv"
    json_path = args.baseline_dir / "step1_baseline.json"
    md_path = args.baseline_dir / "step1_baseline.md"

    df.sort_values("india_auc", ascending=False).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "gates": gates, "combos": df.to_dict(orient="records")}, f, indent=2)
    write_markdown(df, gates, meta, md_path)

    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    country_auc_path = args.comparison_dir / "country_auc_comparison.png"
    gap_path = args.comparison_dir / "india_italy_gap_comparison.png"
    plot_country_auc_comparison(df, country_auc_path)
    plot_gap_comparison(df, gap_path)

    print(f"\n{'=' * 78}\nStep 1 baseline -- {len(df)} combos\n{'=' * 78}")
    print(df[["combo", "india_auc", "india_ci_half_width", "italy_auc", "gap"]].to_string(index=False))
    print("\nGates:")
    for name, gate in sorted(gates.items()):
        print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {name}")
        if not gate["passed"] and name == "5_negative_control" and gate["n_controls_run"] == 0:
            print("         -> run: run_cv_harness.py --combo <name> --shuffle-control within_country")
    print(f"\nWrote {csv_path}\n      {json_path}\n      {md_path}")
    print(f"Wrote {country_auc_path}\n      {gap_path}")

    all_passed = all(g["passed"] for g in gates.values())
    print("\n" + ("STEP 1 CLEAR -- proceed to Step 2." if all_passed else "STEP 1 NOT CLEAR -- do not start Step 2."))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
