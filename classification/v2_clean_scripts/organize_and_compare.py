"""
Organizes a downloaded Kaggle results zip's extracted contents (flat
outputs/{checkpoints,logs,plots}/, per sync_outputs()'s layout) into one
subfolder per (architecture, tissue_type) combo, extracts each combo's
headline metrics, and builds a cross-combo comparison.

Combo names are discovered dynamically from outputs/logs/*_study_summary.json
-- not a hardcoded architecture list -- so this same script works unchanged
for the CNN batch, the transformer batch, or both together if their outputs
ever end up merged in the same directory.

Usage:
    python organize_and_compare.py [outputs_dir]

    outputs_dir defaults to the directory this script lives in, joined with
    "outputs" (i.e. classification/v2_clean_scripts/outputs/, matching where
    the project author has been placing downloaded Kaggle result zips).
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

# The 6 metrics explicitly requested, in this order, read from each combo's
# "overall" bucket in best_val_metrics_by_country. sensitivity IS recall in
# this project's compute_metrics() (verified: same value, kept as two keys
# for readability) -- both are pulled out separately below since both names
# were asked for.
REQUESTED_METRICS = ["f1", "accuracy", "recall", "precision", "specificity", "balanced_accuracy"]


def find_combo_names(logs_dir: Path) -> list:
    return sorted(p.name[: -len("_study_summary.json")] for p in logs_dir.glob("*_study_summary.json"))


def organize_outputs(outputs_dir: Path) -> list:
    """Moves every file for each discovered combo into outputs_dir/{combo_name}/.
    Returns the sorted list of combo names found."""
    checkpoints_dir = outputs_dir / "checkpoints"
    logs_dir = outputs_dir / "logs"
    plots_dir = outputs_dir / "plots"

    if not logs_dir.exists():
        raise FileNotFoundError(f"No logs/ directory under {outputs_dir} -- nothing to organize.")

    combo_names = find_combo_names(logs_dir)
    if not combo_names:
        raise FileNotFoundError(f"No *_study_summary.json files found under {logs_dir}.")

    print(f"Discovered {len(combo_names)} combo(s): {combo_names}\n")

    for combo in combo_names:
        combo_dir = outputs_dir / combo
        combo_dir.mkdir(exist_ok=True)

        ckpt_src = checkpoints_dir / f"best_{combo}.pth"
        if ckpt_src.exists():
            shutil.move(str(ckpt_src), str(combo_dir / ckpt_src.name))
        else:
            print(f"  [{combo}] WARNING: no checkpoint found at {ckpt_src}")

        for log_src in logs_dir.glob(f"{combo}_*"):
            shutil.move(str(log_src), str(combo_dir / log_src.name))

        for plot_src in plots_dir.glob(f"{combo}_*.png"):
            shutil.move(str(plot_src), str(combo_dir / plot_src.name))

        n_files = sum(1 for _ in combo_dir.iterdir())
        print(f"  [{combo}] -> {combo_dir} ({n_files} files)")

    # Clean up now-empty flat dirs
    for d in (checkpoints_dir, logs_dir, plots_dir):
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
            print(f"\nRemoved now-empty {d}")
        elif d.exists():
            remaining = [p.name for p in d.iterdir()]
            print(f"\nNOTE: {d} still has unmatched files, left in place: {remaining}")

    return combo_names


def extract_metrics(outputs_dir: Path, combo: str) -> dict:
    summary_path = outputs_dir / combo / f"{combo}_study_summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    overall = summary["best_val_metrics_by_country"]["overall"]
    india = summary["best_val_metrics_by_country"].get("India", {})
    italy = summary["best_val_metrics_by_country"].get("Italy", {})

    metrics = {
        "model_name": combo,
        "n_trials_run": summary["n_trials_run"],
        "best_trial_number": summary["best_trial_number"],
    }
    for key in REQUESTED_METRICS:
        metrics[key] = overall.get(key)
    # Bonus context beyond the 6 requested metrics -- AUC and the India/Italy
    # split have been the central axis of every other analysis in this
    # project, so included here too rather than silently dropped.
    metrics["auc_overall"] = overall.get("auc")
    metrics["auc_india"] = india.get("auc")
    metrics["auc_italy"] = italy.get("auc")
    if metrics["auc_india"] is not None and metrics["auc_italy"] is not None:
        metrics["india_italy_auc_gap"] = abs(metrics["auc_india"] - metrics["auc_italy"])
    else:
        metrics["india_italy_auc_gap"] = None
    metrics["best_params"] = summary["best_params"]
    metrics["timestamp_utc"] = summary.get("timestamp_utc")

    return metrics


def write_per_model_summary(outputs_dir: Path, combo: str, metrics: dict) -> None:
    combo_dir = outputs_dir / combo

    with open(combo_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        f"# {combo} -- Metrics Summary",
        "",
        f"Best trial: #{metrics['best_trial_number']} (of {metrics['n_trials_run']} run)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in REQUESTED_METRICS:
        val = metrics[key]
        lines.append(f"| {key.replace('_', ' ').title()} | {val:.4f} |" if val is not None else f"| {key} | N/A |")
    lines += [
        f"| AUC (overall) | {metrics['auc_overall']:.4f} |" if metrics["auc_overall"] is not None else "| AUC (overall) | N/A |",
        f"| AUC (India) | {metrics['auc_india']:.4f} |" if metrics["auc_india"] is not None else "| AUC (India) | N/A |",
        f"| AUC (Italy) | {metrics['auc_italy']:.4f} |" if metrics["auc_italy"] is not None else "| AUC (Italy) | N/A |",
        f"| India/Italy AUC gap | {metrics['india_italy_auc_gap']:.4f} |" if metrics["india_italy_auc_gap"] is not None else "| India/Italy AUC gap | N/A |",
        "",
        "## Best hyperparameters",
        "",
        "```json",
        json.dumps(metrics["best_params"], indent=2),
        "```",
    ]
    with open(combo_dir / "metrics_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_comparison(outputs_dir: Path, all_metrics: list) -> None:
    comparison_dir = outputs_dir / "model_comparison"
    comparison_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(all_metrics).drop(columns=["best_params"])
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    csv_path = comparison_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)

    md_cols = ["rank", "model_name"] + REQUESTED_METRICS + ["auc_overall", "india_italy_auc_gap"]
    md_df = df[md_cols].copy()
    for c in REQUESTED_METRICS + ["auc_overall", "india_italy_auc_gap"]:
        md_df[c] = md_df[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    md_df = md_df.rename(columns={
        "model_name": "Model", "f1": "F1", "accuracy": "Accuracy", "recall": "Recall/Sensitivity",
        "precision": "Precision", "specificity": "Specificity", "balanced_accuracy": "Balanced Accuracy",
        "auc_overall": "AUC", "india_italy_auc_gap": "India/Italy AUC Gap",
    })

    def to_markdown_table(frame: pd.DataFrame) -> str:
        header = "| " + " | ".join(frame.columns) + " |"
        sep = "|" + "|".join(["---"] * len(frame.columns)) + "|"
        rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in frame.iterrows()]
        return "\n".join([header, sep] + rows)

    best = df.iloc[0]
    best_conf = df.loc[df["india_italy_auc_gap"].idxmin()] if df["india_italy_auc_gap"].notna().any() else None

    lines = [
        "# CNN Batch -- Model Comparison (clean data, _v2_clean)",
        "",
        f"{len(df)} combos compared, sorted by overall validation F1 (descending).",
        "",
        to_markdown_table(md_df),
        "",
        "## Top performer",
        "",
        f"**{best['model_name']}** -- F1={best['f1']:.4f}, Balanced Accuracy={best['balanced_accuracy']:.4f}, "
        f"AUC={best['auc_overall']:.4f}, India/Italy AUC gap={best['india_italy_auc_gap']:.4f}.",
        "",
    ]
    if best_conf is not None:
        lines += [
            "## Best confound-handling (smallest India/Italy AUC gap)",
            "",
            f"**{best_conf['model_name']}** -- gap={best_conf['india_italy_auc_gap']:.4f}, F1={best_conf['f1']:.4f}.",
            "",
        ]

    with open(comparison_dir / "comparison_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {comparison_dir / 'comparison_report.md'}")
    print(f"\nTop 3 by F1:")
    print(df[["rank", "model_name", "f1", "balanced_accuracy", "auc_overall"]].head(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", nargs="?", default=None,
                         help="Path to the outputs/ directory (default: ./outputs next to this script)")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else Path(__file__).resolve().parent / "outputs"
    print(f"Working on: {outputs_dir}\n")

    combo_names = organize_outputs(outputs_dir)

    print("\nExtracting metrics...")
    all_metrics = []
    for combo in combo_names:
        metrics = extract_metrics(outputs_dir, combo)
        write_per_model_summary(outputs_dir, combo, metrics)
        all_metrics.append(metrics)
        print(f"  [{combo}] F1={metrics['f1']:.4f}  wrote metrics_summary.json + .md")

    print("\nBuilding comparison...")
    build_comparison(outputs_dir, all_metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
