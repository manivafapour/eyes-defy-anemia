"""
Plotting utilities for the segmentation pipeline -- one PNG per logical
group of metrics, generated once at the end of run_study() (never per
epoch/per trial) and saved to Segmentation/outputs/plots/, matching this
project's outputs/logs/ convention of "small enough to keep in version
control." Added alongside the raw JSON/CSV numbers per the project
author's explicit request for Kaggle output to include plots, not just
values, for every metric.

Plotting is scoped strictly to the STUDY'S BEST TRIAL for the training
curves (matches classification's own established convention: never plot
every trial, only the winner) and to the final TEST-set evaluation for
everything else, since that's the bias-free, "real" result -- not the
validation-set numbers used to pick the winning trial.
"""

import matplotlib

matplotlib.use("Agg")  # headless -- this module runs during training scripts, not interactive notebooks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(epoch_history: pd.DataFrame, model_name: str, out_path) -> None:
    """Two panels for the best trial only: (1) train vs val loss per epoch,
    (2) val Dice/IoU/Precision/Recall per epoch -- shows whether the model
    was still improving when early stopping triggered, and whether Dice
    gains came with a precision/recall tradeoff or an improvement on both."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epoch_history["epoch"], epoch_history["train_loss"], label="train_loss", marker="o")
    axes[0].plot(epoch_history["epoch"], epoch_history["val_loss"], label="val_loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name}\nTrain vs. validation loss (best trial)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for col, label in [("val_dice", "Dice"), ("val_iou", "IoU"), ("val_precision", "Precision"), ("val_recall", "Recall")]:
        axes[1].plot(epoch_history["epoch"], epoch_history[col], label=label, marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"{model_name}\nValidation metrics (best trial)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_dice_iou_curve(epoch_history: pd.DataFrame, model_name: str, out_path) -> None:
    """Dedicated Dice-vs-epoch / IoU-vs-epoch line chart, best trial only --
    the standard headline figure in segmentation papers, kept separate from
    plot_training_curves' combined 4-metric panel (which also carries
    precision/recall) specifically so Dice/IoU convergence is readable on
    its own, without three other lines competing for the same axes.
    Marks the best (max-Dice) epoch explicitly, since that's the epoch the
    saved checkpoint actually came from -- not necessarily the last one, if
    early stopping's patience let training continue a few epochs past it."""
    best_idx = epoch_history["val_dice"].idxmax()
    best_epoch = epoch_history.loc[best_idx, "epoch"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_history["epoch"], epoch_history["val_dice"], label="Dice", marker="o", color="tab:blue")
    ax.plot(epoch_history["epoch"], epoch_history["val_iou"], label="IoU", marker="o", color="tab:orange")
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch}, checkpoint saved)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model_name}\nValidation Dice & IoU vs. epoch (best trial)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_loss_fn_comparison(trials_df: pd.DataFrame, model_name: str, out_path) -> None:
    """Mean and best validation Dice achieved per loss_fn candidate
    (bce_dice vs. focal_tversky) across all trials in the study -- lets you
    see at a glance whether one loss function systematically outperformed
    the other for this architecture, not just which trial happened to win."""
    if "params_loss_fn" not in trials_df.columns:
        return

    grouped = trials_df.groupby("params_loss_fn")["value"].agg(mean="mean", best="max")

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(grouped))
    width = 0.35
    ax.bar(x - width / 2, grouped["mean"], width, label="Mean Dice")
    ax.bar(x + width / 2, grouped["best"], width, label="Best Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_ylabel("Validation Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model_name}\nLoss function comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_test_set_summary(test_metrics_summary: dict, model_name: str, out_path) -> None:
    """Final TEST-set metrics (not validation) -- the bias-free numbers.
    Two panels since HD95 is in pixels, not the 0-1 range the other four
    metrics share, and putting it on the same axis would flatten the
    other four bars to invisibility."""
    overall = test_metrics_summary["overall"]
    unit_metrics = ["dice", "iou", "precision", "recall"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(unit_metrics, [overall[m] for m in unit_metrics], color="steelblue")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"{model_name}\nTest-set metrics (0-1 scale)")
    axes[0].grid(alpha=0.3, axis="y")

    n_undefined = test_metrics_summary["n_hd95_undefined_empty_mask"]
    n_total = test_metrics_summary["n_test_patients"]
    axes[1].bar(["HD95"], [overall["hd95"]], color="indianred")
    axes[1].set_ylabel("Pixels")
    axes[1].set_title(f"Hausdorff Distance (95th pct)\n(undefined for {n_undefined}/{n_total} patients)")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_test_set_by_country(test_metrics_summary: dict, model_name: str, out_path) -> None:
    """Same test-set metrics, grouped by country -- ties into this
    project's documented India/Italy demographic confound (CLAUDE.md Sec
    0.5): a model whose segmentation quality quietly differs by country is
    exactly the kind of thing an aggregate-only summary would hide."""
    by_country = test_metrics_summary["by_country"]
    countries = list(by_country.keys())
    unit_metrics = ["dice", "iou", "precision", "recall"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(unit_metrics))
    width = 0.8 / max(len(countries), 1)
    for i, country in enumerate(countries):
        values = [by_country[country][m] for m in unit_metrics]
        axes[0].bar(x + i * width, values, width, label=country)
    axes[0].set_xticks(x + width * (len(countries) - 1) / 2)
    axes[0].set_xticklabels(unit_metrics)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"{model_name}\nTest-set metrics by country")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(countries, [by_country[c]["hd95"] for c in countries], color="indianred")
    axes[1].set_ylabel("Pixels")
    axes[1].set_title("HD95 by country")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_patient_distribution(test_df: pd.DataFrame, model_name: str, out_path) -> None:
    """Per-patient Dice spread (box + individual points, colored by
    country) -- a mean hides variance; this shows whether the model is
    consistently mediocre or excellent-on-most-but-terrible-on-a-few, which
    a single aggregate number can't distinguish and which matters a lot
    for a thesis claim about reliability, not just average quality."""
    fig, ax = plt.subplots(figsize=(7, 5))

    countries = sorted(test_df["country"].unique())
    data = [test_df.loc[test_df["country"] == c, "dice"] for c in countries]

    # Setting tick labels via set_xticklabels() rather than boxplot()'s own
    # labels/tick_labels kwarg -- that kwarg was renamed between matplotlib
    # versions (labels -> tick_labels, removed entirely in 3.11 here vs.
    # whatever version Kaggle's base image ships), so this is the one
    # spelling guaranteed to work across both without knowing either
    # version in advance.
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(countries) + 1))
    ax.set_xticklabels(countries)
    for i, c in enumerate(countries, start=1):
        y = test_df.loc[test_df["country"] == c, "dice"]
        x = np.random.default_rng(42).normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, s=20, color="steelblue")

    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model_name}\nPer-patient test Dice distribution (n={len(test_df)})")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_all_plots(
    plots_dir,
    model_name: str,
    epoch_history: pd.DataFrame,
    trials_df: pd.DataFrame,
    test_metrics_summary: dict = None,
    test_df: pd.DataFrame = None,
) -> list:
    """Generates every plot this module knows how to make for one model's
    completed study, skipping anything whose required data isn't available
    (e.g. no test_metrics_summary if no checkpoint was ever saved) rather
    than erroring. Returns the list of paths actually written."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    if epoch_history is not None and len(epoch_history):
        path = plots_dir / f"{model_name}_training_curves.png"
        plot_training_curves(epoch_history, model_name, path)
        written.append(path)

        path = plots_dir / f"{model_name}_dice_iou_curve.png"
        plot_dice_iou_curve(epoch_history, model_name, path)
        written.append(path)

    loss_fn_path = plots_dir / f"{model_name}_loss_fn_comparison.png"
    plot_loss_fn_comparison(trials_df, model_name, loss_fn_path)
    if loss_fn_path.exists():
        written.append(loss_fn_path)

    if test_metrics_summary is not None:
        path = plots_dir / f"{model_name}_test_summary.png"
        plot_test_set_summary(test_metrics_summary, model_name, path)
        written.append(path)

        path = plots_dir / f"{model_name}_test_by_country.png"
        plot_test_set_by_country(test_metrics_summary, model_name, path)
        written.append(path)

    if test_df is not None and len(test_df):
        path = plots_dir / f"{model_name}_test_per_patient_dice.png"
        plot_per_patient_distribution(test_df, model_name, path)
        written.append(path)

    return written
