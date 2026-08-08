"""
Step 1 -- Measurement Harness: per-fold and cross-combo diagnostic plots.

Added after the first CNN run completed: the original design deliberately
saved no plots (cv_config.py, "no per-fold checkpoints... Step 1 needs
predictions, not weights") on the reasoning that only the pooled statistic
mattered. That left no way to visually inspect any individual fold's
training behavior or spot-check a fold's own confusion matrix/ROC curve --
useful for a thesis figure and for sanity-checking that nothing degenerate
happened on a specific fold, even though it isn't the headline number.

Every per-combo plot here is laid out as an n_repeats x n_splits grid (5x5
for the default design) so every fold can be visually compared to every
other fold in ONE image, per plot type, per combo -- rather than 25 separate
tiny files. Purely a visualization layer: everything plotted here was
already computed by cv_engine.py (loss histories) or cv_stats.py
(fold_level_metrics, bootstrap CIs) -- no numbers are computed in this file.
"""

import matplotlib

matplotlib.use("Agg")  # headless -- runs on Kaggle containers, no display available
import matplotlib.pyplot as plt
import numpy as np


def _grid_dims(fold_keys) -> tuple:
    repeats = sorted({r for r, _f in fold_keys})
    folds = sorted({f for _r, f in fold_keys})
    return repeats, folds


def plot_loss_curves_grid(combo_name: str, histories: dict, out_path) -> str:
    """histories: {(repeat, fold): {"train_loss": [...], "inner_val_loss": [...]}}.
    One subplot per fold, train vs. inner-validation loss over epochs --
    inner-validation, not the outer held-out fold, since that's the loss the
    early-stopping decision was actually based on (cv_engine.py)."""
    repeats, folds = _grid_dims(histories.keys())
    fig, axes = plt.subplots(
        len(repeats), len(folds), figsize=(3.6 * len(folds), 2.8 * len(repeats)), squeeze=False
    )
    for i, r in enumerate(repeats):
        for j, f in enumerate(folds):
            ax = axes[i][j]
            h = histories.get((r, f))
            if h is None:
                ax.axis("off")
                continue
            epochs = range(1, len(h["train_loss"]) + 1)
            ax.plot(epochs, h["train_loss"], label="train", linewidth=1)
            ax.plot(epochs, h["inner_val_loss"], label="inner val", linewidth=1)
            ax.set_title(f"repeat {r}, fold {f}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(f"{combo_name} -- inner train/val loss per fold", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_confusion_matrices_grid(combo_name: str, per_fold_metrics: dict, out_path) -> str:
    """per_fold_metrics: {(repeat, fold): dict from cv_stats.fold_level_metrics}.
    Each fold's OWN held-out patients (~37), not the pooled 184 -- these are
    per-fold diagnostics, not the headline pooled result."""
    repeats, folds = _grid_dims(per_fold_metrics.keys())
    fig, axes = plt.subplots(
        len(repeats), len(folds), figsize=(2.6 * len(folds), 2.6 * len(repeats)), squeeze=False
    )
    for i, r in enumerate(repeats):
        for j, f in enumerate(folds):
            ax = axes[i][j]
            m = per_fold_metrics.get((r, f))
            cm = m.get("confusion_matrix") if m else None
            if cm is None:
                ax.set_title(f"r{r} f{f}\nno data", fontsize=7)
                ax.axis("off")
                continue
            cm = np.array(cm)
            ax.imshow(cm, cmap="Blues", vmin=0)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, str(cm[a, b]), ha="center", va="center", fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Non-an.", "Anemic"], fontsize=6)
            ax.set_yticklabels(["Non-an.", "Anemic"], fontsize=6)
            ax.set_title(f"repeat {r}, fold {f} (n={m['n']})", fontsize=7)
    fig.suptitle(f"{combo_name} -- confusion matrix per fold (x=predicted, y=true)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_roc_curves_grid(combo_name: str, per_fold_metrics: dict, out_path) -> str:
    repeats, folds = _grid_dims(per_fold_metrics.keys())
    fig, axes = plt.subplots(
        len(repeats), len(folds), figsize=(3.0 * len(folds), 2.8 * len(repeats)), squeeze=False
    )
    for i, r in enumerate(repeats):
        for j, f in enumerate(folds):
            ax = axes[i][j]
            m = per_fold_metrics.get((r, f))
            roc = m.get("roc_curve") if m else None
            if roc is None:
                ax.set_title(f"r{r} f{f}\nsingle class in fold", fontsize=7)
                ax.axis("off")
                continue
            ax.plot(roc["fpr"], roc["tpr"], linewidth=1.2)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.7)
            ax.set_title(f"repeat {r}, fold {f}  AUC={m['auc']:.2f}", fontsize=7.5)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{combo_name} -- ROC curve per fold (own held-out set, not pooled)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_fold_metrics_summary(combo_name: str, fold_metrics_df, out_path) -> str:
    """One combined line/marker plot of AUC / F1 / Balanced Accuracy across
    every (repeat, fold) -- the 'overall evaluation for each fold' view, so
    fold-to-fold consistency (or a specific fold going badly wrong) is
    visible at a glance rather than only in the CSV table."""
    df = fold_metrics_df.sort_values(["repeat", "fold"]).reset_index(drop=True)
    x = range(len(df))
    tick_labels = [f"r{r}f{f}" for r, f in zip(df["repeat"], df["fold"])]

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(df)), 5))
    for col, marker in [("auc", "o"), ("f1", "s"), ("balanced_accuracy", "^")]:
        vals = df[col]
        ax.plot(x, vals, marker=marker, markersize=4, linewidth=1, label=col.replace("_", " "))
    ax.set_xticks(list(x))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("score")
    ax.set_title(f"{combo_name} -- per-fold evaluation summary (own held-out set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_country_auc_comparison(df, out_path) -> str:
    """Cross-combo comparison: pooled India/Italy/overall AUC with bootstrap
    CI whiskers, one panel each, all sorted by India AUC descending so the
    same combo ordering holds across all three panels for easy cross-reference."""
    order = df.sort_values("india_auc", ascending=False)["combo"].tolist()
    ordered = df.set_index("combo").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, 0.32 * len(ordered))), sharey=True)
    panels = [
        ("india_auc", "india_ci_low", "india_ci_high", "India AUC"),
        ("italy_auc", "italy_ci_low", "italy_ci_high", "Italy AUC"),
        ("overall_auc", "overall_ci_low", "overall_ci_high", "Overall AUC"),
    ]
    y = np.arange(len(ordered))
    for ax, (val_col, lo_col, hi_col, title) in zip(axes, panels):
        vals = ordered[val_col].to_numpy(dtype=float)
        lo = ordered[lo_col].to_numpy(dtype=float)
        hi = ordered[hi_col].to_numpy(dtype=float)
        xerr = np.vstack([vals - lo, hi - vals])
        ax.barh(y, vals, xerr=xerr, capsize=3, color="#4C72B0", ecolor="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels(ordered["combo"], fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.invert_yaxis()
    fig.suptitle("Pooled out-of-fold AUC by country, with 95% bootstrap CI", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_gap_comparison(df, out_path) -> str:
    """India-Italy AUC gap per combo, with bootstrap CI whiskers, sorted
    ascending (smallest/best-confound-handling first)."""
    ordered = df.sort_values("gap", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    vals = ordered["gap"].to_numpy(dtype=float)
    lo = ordered["gap_ci_low"].to_numpy(dtype=float)
    hi = ordered["gap_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([vals - lo, hi - vals])

    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(ordered))))
    colors = ["#55A868" if excl else "#C44E52" for excl in ordered["gap_ci_excludes_zero"]]
    ax.barh(y, vals, xerr=xerr, capsize=3, color=colors, ecolor="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["combo"], fontsize=8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("India AUC - Italy AUC (pooled)")
    ax.set_title("India/Italy AUC gap, 95% bootstrap CI (green = CI excludes 0)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
