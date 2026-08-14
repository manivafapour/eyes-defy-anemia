"""
Shared, architecture-agnostic pieces reused across every per-model partial
fine-tuning engine in this directory (finetune_engine.py for ConvNeXt-Base,
finetune_engine_coatnet3.py for CoAtNet-3, and any future ones) -- the
train loop and plotting functions are 100% identical logic regardless of
which backbone or which submodule gets unfrozen, so they live here once
instead of being copy-pasted per model. What stays per-model, deliberately
NOT unified here: build_finetune_model() (which submodules to unfreeze is
genuinely architecture-specific) and the top-level run_finetune()
orchestration (differs enough in per-model hyperparameter/param-group
setup that forcing it into one generic function isn't worth the
indirection at just 2-3 models -- reconsider if a 4th one is added).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def freeze_batchnorm_outside(model: torch.nn.Module, trainable_root: torch.nn.Module) -> None:
    """Call every time after model.train() when the model has any BatchNorm
    layers outside the region being fine-tuned (EfficientNet-B3/RegNetY-16GF;
    not needed for ConvNeXt-Base or CoAtNet-3's attention stages, which have
    none). requires_grad=False stops the optimizer from updating a
    BatchNorm's weight/bias, but its running_mean/running_var are BUFFERS,
    not parameters -- they keep updating on every forward pass in .train()
    mode regardless of requires_grad. Left unhandled, every "frozen"
    BatchNorm layer in the network would silently drift its running
    statistics based on this project's small per-epoch batches, undermining
    the entire "same frozen backbone, only these params changed" premise
    the before/after comparison depends on. Related-but-distinct from
    classification/step1_cv_harness/cv_engine.py's _mutable_state() (which
    solves a checkpoint-snapshotting problem, not a drift-prevention one) --
    that precedent is what flagged buffers as the thing to watch for."""
    trainable_bn_ids = {id(m) for m in trainable_root.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and id(m) not in trainable_bn_ids:
            m.eval()


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_max_norm: float) -> float:
    model.train()
    running_loss = 0.0
    for images, labels, _countries in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def plot_loss_curve(model_name: str, plots_dir: Path, train_loss_history, val_loss_history) -> Path:
    epochs = range(1, len(train_loss_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss_history, label="train loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss_history, label="val loss", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title(f"{model_name} -- train vs val loss")
    ax.legend()
    ax.grid(alpha=0.3)
    path = plots_dir / f"{model_name}_loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lr_curve(model_name: str, plots_dir: Path, lr_histories: dict) -> Path:
    """lr_histories: {param_group_label: [lr per epoch]} -- generalizes to
    any number of discriminative-LR param groups, not just 2."""
    epochs = range(1, len(next(iter(lr_histories.values()))) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, history in lr_histories.items():
        ax.plot(epochs, history, marker="o", markersize=3, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title(f"{model_name} -- discriminative LR schedule")
    ax.legend()
    ax.grid(alpha=0.3)
    path = plots_dir / f"{model_name}_lr_schedule.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_val_metrics_curve(model_name: str, plots_dir: Path, val_overall_metrics_history) -> Path:
    epochs = range(1, len(val_overall_metrics_history) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, marker in [("accuracy", "o"), ("f1", "s"), ("sensitivity", "^"), ("specificity", "v")]:
        values = [m[key] for m in val_overall_metrics_history]
        if any(v is None for v in values):
            continue
        ax.plot(epochs, values, label=key, marker=marker, markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{model_name} -- validation metrics over epochs")
    ax.legend()
    ax.grid(alpha=0.3)
    path = plots_dir / f"{model_name}_val_metrics_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrices(model_name: str, plots_dir: Path, best_val_metrics: dict) -> Path:
    buckets = ["overall", "India", "Italy"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, bucket in zip(axes, buckets):
        cm = best_val_metrics[bucket].get("confusion_matrix")
        n = best_val_metrics[bucket]["n"]
        if cm is None:
            ax.set_title(f"{bucket} (n={n})\nno data")
            ax.axis("off")
            continue
        cm = np.array(cm)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-anemic", "Anemic"])
        ax.set_yticklabels(["Non-anemic", "Anemic"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{bucket} (n={n})")
    fig.suptitle(f"{model_name} -- stratified confusion matrix (best val epoch)")
    fig.tight_layout()
    path = plots_dir / f"{model_name}_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(model_name: str, plots_dir: Path, best_val_metrics: dict) -> Path | None:
    buckets = ["overall", "India", "Italy"]
    overall_roc = best_val_metrics["overall"].get("roc_curve")
    if overall_roc is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, bucket in zip(axes, buckets):
        roc = best_val_metrics[bucket].get("roc_curve")
        auc = best_val_metrics[bucket].get("auc")
        n = best_val_metrics[bucket]["n"]
        if roc is None:
            ax.set_title(f"{bucket} (n={n})\nonly one class present")
            ax.axis("off")
            continue
        ax.plot(roc["fpr"], roc["tpr"], label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{bucket} (n={n})")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model_name} -- stratified ROC curve (best val epoch)")
    fig.tight_layout()
    path = plots_dir / f"{model_name}_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_before_after(model_name: str, plots_dir: Path, baseline_test_metrics: dict, finetuned_test_metrics: dict) -> Path:
    """The key deliverable plot: frozen-backbone baseline vs fine-tuned, on
    the sealed test set -- overall F1/Accuracy/AUC, and India/Italy AUC
    side by side so a widening confound gap is immediately visible."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    labels = ["F1", "Accuracy", "AUC"]
    baseline_vals = [baseline_test_metrics["overall"]["f1"], baseline_test_metrics["overall"]["accuracy"], baseline_test_metrics["overall"].get("auc") or 0]
    finetuned_vals = [finetuned_test_metrics["overall"]["f1"], finetuned_test_metrics["overall"]["accuracy"], finetuned_test_metrics["overall"].get("auc") or 0]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, baseline_vals, width, label="frozen backbone (baseline)", color="tab:gray")
    ax.bar(x + width / 2, finetuned_vals, width, label="fine-tuned", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Test set: overall")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    countries = ["India", "Italy"]
    baseline_auc = [baseline_test_metrics[c].get("auc") or 0 for c in countries]
    finetuned_auc = [finetuned_test_metrics[c].get("auc") or 0 for c in countries]
    x2 = np.arange(len(countries))
    ax2.bar(x2 - width / 2, baseline_auc, width, label="frozen backbone (baseline)", color="tab:gray")
    ax2.bar(x2 + width / 2, finetuned_auc, width, label="fine-tuned", color="tab:orange")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(countries)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Test set: AUC by country (confound check)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{model_name} -- before vs after fine-tuning")
    fig.tight_layout()
    path = plots_dir / f"{model_name}_before_after.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved before/after comparison plot to {path}")
    return path


def print_before_after(baseline: dict, finetuned: dict) -> None:
    if baseline is None or finetuned is None:
        return
    print("\n=== Before (frozen backbone) vs after (fine-tuned) -- sealed TEST set ===")
    for bucket in ["overall", "India", "Italy"]:
        b, f = baseline[bucket], finetuned[bucket]
        print(
            f"  {bucket:8s}  F1: {b['f1']:.4f} -> {f['f1']:.4f}   "
            f"Acc: {b['accuracy']:.4f} -> {f['accuracy']:.4f}   "
            f"AUC: {b.get('auc')} -> {f.get('auc')}"
        )
    b_gap = abs(baseline["India"].get("auc", 0) - baseline["Italy"].get("auc", 0)) if baseline["India"].get("auc") is not None and baseline["Italy"].get("auc") is not None else None
    f_gap = abs(finetuned["India"].get("auc", 0) - finetuned["Italy"].get("auc", 0)) if finetuned["India"].get("auc") is not None and finetuned["Italy"].get("auc") is not None else None
    print(f"  India/Italy AUC gap: {b_gap} -> {f_gap}  ({'WIDER -- shortcut risk' if (b_gap is not None and f_gap is not None and f_gap > b_gap) else 'not wider'})")


# --------------------------------------------------------------------------
# Cross-model comparison (batch 2 of the fine-tune pilot programme) -- takes
# every model's own *_history.json (written by each engine's run_finetune(),
# read back rather than passed in-memory so the comparison step works
# identically whether a model was just trained in this notebook run or was
# already on disk from an earlier session) and produces one combined view
# across N models, generalizing plot_before_after (which only ever compared
# one model's own before/after) to a cross-model lens.
# --------------------------------------------------------------------------
def load_finetune_history(logs_dir: Path, model_name: str) -> dict | None:
    """Reads {model_name}_history.json back off disk. Returns None (not a
    raised error) if a model hasn't been run yet in this environment --
    callers building a partial comparison (e.g. only some models trained
    so far) can filter these out rather than crash."""
    path = Path(logs_dir) / f"{model_name}_history.json"
    if not path.exists():
        print(f"[load_finetune_history] {path} not found -- skipping {model_name}")
        return None
    with open(path) as f:
        return json.load(f)


def plot_cross_model_val_f1_comparison(model_histories: dict, plots_dir: Path) -> Path:
    """model_histories: {model_name: history_dict} (as loaded by
    load_finetune_history). Compares baseline_val_metrics['overall']['f1']
    (the frozen-backbone starting point) against best_val_f1 (the metric
    that actually drove checkpoint selection during fine-tuning) for every
    model side by side."""
    names = list(model_histories.keys())
    baseline_f1 = [model_histories[n]["baseline_val_metrics"]["overall"]["f1"] for n in names]
    finetuned_f1 = [model_histories[n]["best_val_f1"] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, baseline_f1, width, label="frozen backbone (baseline)", color="tab:gray")
    ax.bar(x + width / 2, finetuned_f1, width, label="fine-tuned (best epoch)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Validation F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-model: validation F1, frozen baseline vs fine-tuned")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "cross_model_val_f1_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cross_model_test_metrics_comparison(model_histories: dict, plots_dir: Path) -> Path:
    """One panel per metric (F1, Accuracy, AUC), each showing every model's
    baseline vs fine-tuned sealed-TEST score -- the multi-model analog of
    plot_before_after's left panel."""
    names = list(model_histories.keys())
    metrics = ["f1", "accuracy", "auc"]
    fig, axes = plt.subplots(1, 3, figsize=(max(15, 3.2 * len(names)), 5))
    x = np.arange(len(names))
    width = 0.35
    for ax, metric in zip(axes, metrics):
        baseline_vals = [model_histories[n]["baseline_test_metrics"]["overall"].get(metric) or 0 for n in names]
        finetuned_vals = [model_histories[n]["test_metrics"]["overall"].get(metric) or 0 for n in names]
        ax.bar(x - width / 2, baseline_vals, width, label="baseline", color="tab:gray")
        ax.bar(x + width / 2, finetuned_vals, width, label="fine-tuned", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Test {metric.upper()}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    axes[0].set_ylabel("score")
    fig.suptitle("Cross-model: sealed TEST set, frozen baseline vs fine-tuned")
    fig.tight_layout()
    path = plots_dir / "cross_model_test_metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cross_model_auc_gap_comparison(model_histories: dict, plots_dir: Path) -> Path:
    """India/Italy AUC gap, baseline vs fine-tuned, across every model --
    the multi-model analog of plot_before_after's right panel (the confound
    check this whole programme has tracked per-model from the start)."""
    names = list(model_histories.keys())

    def gap(metrics):
        india_auc = metrics["India"].get("auc")
        italy_auc = metrics["Italy"].get("auc")
        if india_auc is None or italy_auc is None:
            return None
        return abs(india_auc - italy_auc)

    baseline_gaps = [gap(model_histories[n]["baseline_test_metrics"]) or 0 for n in names]
    finetuned_gaps = [gap(model_histories[n]["test_metrics"]) or 0 for n in names]

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, baseline_gaps, width, label="baseline", color="tab:gray")
    ax.bar(x + width / 2, finetuned_gaps, width, label="fine-tuned", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("|India AUC - Italy AUC|")
    ax.set_title("Cross-model: India/Italy AUC gap (confound check), baseline vs fine-tuned")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = plots_dir / "cross_model_auc_gap_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def build_cross_model_summary_table(model_histories: dict, logs_dir: Path) -> tuple[Path, Path]:
    """Writes both a full-detail CSV and a headline-columns markdown table,
    same convention as classification/outputs/v2_comparison_results/
    (comparison_table.csv + comparison_table.md) elsewhere in this project."""
    import csv

    rows = []
    for name, h in model_histories.items():
        b_test, f_test = h["baseline_test_metrics"], h["test_metrics"]
        b_india_auc, f_india_auc = b_test["India"].get("auc"), f_test["India"].get("auc")
        b_italy_auc, f_italy_auc = b_test["Italy"].get("auc"), f_test["Italy"].get("auc")
        b_gap = abs(b_india_auc - b_italy_auc) if b_india_auc is not None and b_italy_auc is not None else None
        f_gap = abs(f_india_auc - f_italy_auc) if f_india_auc is not None and f_italy_auc is not None else None
        rows.append(
            {
                "model": name,
                "baseline_val_f1": h["baseline_val_metrics"]["overall"]["f1"],
                "finetuned_best_val_f1": h["best_val_f1"],
                "baseline_test_f1": b_test["overall"]["f1"],
                "finetuned_test_f1": f_test["overall"]["f1"],
                "baseline_test_accuracy": b_test["overall"]["accuracy"],
                "finetuned_test_accuracy": f_test["overall"]["accuracy"],
                "baseline_test_auc": b_test["overall"].get("auc"),
                "finetuned_test_auc": f_test["overall"].get("auc"),
                "baseline_india_auc": b_india_auc,
                "finetuned_india_auc": f_india_auc,
                "baseline_italy_auc": b_italy_auc,
                "finetuned_italy_auc": f_italy_auc,
                "baseline_auc_gap": b_gap,
                "finetuned_auc_gap": f_gap,
                "n_epochs_run": h["n_epochs_run"],
            }
        )

    csv_path = Path(logs_dir) / "cross_model_comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(logs_dir) / "cross_model_comparison_table.md"
    with open(md_path, "w") as f:
        f.write("| Model | Baseline val F1 | Fine-tuned val F1 | Test F1 (base->ft) | Test AUC (base->ft) | India AUC (base->ft) | Italy AUC (base->ft) | AUC gap (base->ft) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['model']} | {r['baseline_val_f1']:.4f} | {r['finetuned_best_val_f1']:.4f} | "
                f"{r['baseline_test_f1']:.4f}->{r['finetuned_test_f1']:.4f} | "
                f"{r['baseline_test_auc']}->{r['finetuned_test_auc']} | "
                f"{r['baseline_india_auc']}->{r['finetuned_india_auc']} | "
                f"{r['baseline_italy_auc']}->{r['finetuned_italy_auc']} | "
                f"{r['baseline_auc_gap']}->{r['finetuned_auc_gap']} |\n"
            )
    print(f"Saved cross-model comparison table to {csv_path} and {md_path}")
    return csv_path, md_path


def plot_cross_model_comparison(model_histories: dict, plots_dir: Path, logs_dir: Path) -> dict:
    """Top-level entry point: given {model_name: history_dict} for every
    model to compare (2+), produces all 3 comparison plots plus the summary
    table in one call. Skips models with a missing test_metrics/best
    checkpoint (mirrors plot_before_after's own None-guard) rather than
    crashing the whole comparison over one incomplete run."""
    complete = {n: h for n, h in model_histories.items() if h is not None and h.get("test_metrics") is not None}
    missing = set(model_histories) - set(complete)
    if missing:
        print(f"[plot_cross_model_comparison] Skipping incomplete/missing models: {sorted(missing)}")
    if len(complete) < 2:
        print(f"[plot_cross_model_comparison] Need at least 2 complete models to compare, got {len(complete)} -- skipping.")
        return {}

    paths = {
        "val_f1": plot_cross_model_val_f1_comparison(complete, plots_dir),
        "test_metrics": plot_cross_model_test_metrics_comparison(complete, plots_dir),
        "auc_gap": plot_cross_model_auc_gap_comparison(complete, plots_dir),
    }
    paths["csv_table"], paths["md_table"] = build_cross_model_summary_table(complete, logs_dir)
    print(f"Cross-model comparison built for {len(complete)} models: {sorted(complete)}")
    return paths
