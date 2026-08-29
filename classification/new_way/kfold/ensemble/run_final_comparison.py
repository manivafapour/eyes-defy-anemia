"""
Entry point: combines every ensembling technique tried in this folder --
soft voting / uniform soup / greedy soup / stacking (ensemble_engine.py,
via run_ensemble.py's saved ensemble_comparison.json) and bagging /
boosting (bagging_boosting.py, via run_bagging_boosting.py's saved
bagging_boosting_summary.json) -- into one master table and plot, sorted by
test F1, with an explicit leak-free/leaky annotation per row so the "best"
number is never read without its caveat attached.

Requires both run_ensemble.py and run_bagging_boosting.py to have already
been run at least once (reads their saved JSON outputs; does not
recompute anything itself).

Standalone-runnable: `python classification/new_way/kfold/ensemble/run_final_comparison.py`
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ENSEMBLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENSEMBLE_DIR))

from bagging_boosting import BB_CACHE_DIR  # noqa: E402
from ensemble_engine import LOGS_DIR, PLOTS_DIR  # noqa: E402

KIND_NOTES = {
    "baseline": "leak-free (single trained checkpoint)",
    "ensemble": "leak-free (soft-voting average of fixed checkpoints)",
    "soup": "leak-free (weight-averaged, fixed fold set)",
    "greedy_soup": "OPTIMISTIC -- fold subset chosen by peeking at this same test set",
    "stacking": "leak-free (meta-learner trained on out-of-fold predictions only)",
    "bagging": "leak-free (bootstrap-trained heads, sealed test set untouched during training)",
    "boosting": "leak-free (sequentially-trained heads, sealed test set untouched during training)",
}


def load_ensemble_engine_rows() -> list:
    path = LOGS_DIR / "ensemble_comparison.csv"
    if not path.exists():
        print(f"  (skipping -- {path} not found; run run_ensemble.py first)")
        return []
    df = pd.read_csv(path)
    return df.to_dict("records")


def load_bagging_boosting_rows() -> list:
    path = BB_CACHE_DIR / "bagging_boosting_summary.json"
    if not path.exists():
        print(f"  (skipping -- {path} not found; run run_bagging_boosting.py first)")
        return []
    with open(path) as f:
        data = json.load(f)

    rows = []
    for technique in ["bagging", "boosting"]:
        for model_key, r in data.get(technique, {}).items():
            m = r["final_metrics"]["overall"]
            n_members = r.get("n_bags") or r.get("n_rounds_used_in_final")
            rows.append(
                {
                    "name": f"{model_key}_{technique}",
                    "kind": technique,
                    "n_members": n_members,
                    "f1": m["f1"],
                    "accuracy": m["accuracy"],
                    "auc": m["auc"],
                    "sensitivity": m.get("sensitivity"),
                    "specificity": m.get("specificity"),
                }
            )
    return rows


if __name__ == "__main__":
    rows = load_ensemble_engine_rows() + load_bagging_boosting_rows()
    if not rows:
        raise SystemExit("No results found -- run run_ensemble.py and run_bagging_boosting.py first.")

    df = pd.DataFrame(rows)
    df["note"] = df["kind"].map(KIND_NOTES)
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)

    csv_path = LOGS_DIR / "all_ensemble_techniques_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    print("\n=== All techniques, sorted by test F1 ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 200):
        print(df[["name", "kind", "n_members", "f1", "accuracy", "auc"]].to_string(index=False))

    print("\n=== Best result EXCLUDING the leaky greedy_soup kind ===")
    clean = df[df["kind"] != "greedy_soup"]
    print(clean.iloc[0][["name", "kind", "f1", "accuracy", "auc"]])

    kind_colors = {
        "baseline": "tab:blue",
        "ensemble": "tab:orange",
        "soup": "tab:green",
        "greedy_soup": "tab:red",
        "stacking": "tab:purple",
        "bagging": "tab:brown",
        "boosting": "tab:pink",
    }
    plot_df = df.sort_values(["kind", "f1"], ascending=[True, False]).reset_index(drop=True)
    colors = [kind_colors.get(k, "gray") for k in plot_df["kind"]]

    fig, ax = plt.subplots(figsize=(max(14, 0.35 * len(plot_df)), 7))
    x = range(len(plot_df))
    ax.bar(x, plot_df["f1"], color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["name"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Test F1 (overall)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "All ensembling techniques -- sealed test set\n"
        "blue=single checkpoint / orange=soft-vote / green=soup / red=greedy soup (OPTIMISTIC) / "
        "purple=stacking / brown=bagging / pink=boosting"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = PLOTS_DIR / "all_ensemble_techniques_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")
