"""
Shared evaluation-metric utilities for the segmentation pipeline, used by
both trainer_engine.py's per-epoch validation loop (cheap, vectorized
metrics only -- Dice/IoU/Precision/Recall) and the one-time final test-set
evaluation run at the end of run_study() (adds the more expensive per-image
Hausdorff Distance, and produces a per-patient breakdown suitable for a
country-stratified summary or a paired significance test against another
model -- see compare_models_significance.py).

Added in response to the project author's explicit request for a broader
evaluation-metric set to defend in the thesis, beyond the original Dice/IoU
pair: precision/recall (decomposes Dice into its false-positive/
false-negative components, directly interpretable against the
FocalTverskyLoss alpha/beta design in CLAUDE.md Sec 3.2b), and Hausdorff
Distance (HD95) -- a boundary-based metric that a region-overlap metric
like Dice/IoU cannot capture (two masks with identical Dice can have very
different edge precision), and the second metric TransUNet's own paper
reports alongside Dice, making this directly comparable to that
architecture's original published results.
"""

import torch
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt


# --------------------------------------------------------------------------
# Cheap, batch-vectorized metrics -- safe to compute every validation epoch
# --------------------------------------------------------------------------
def compute_batch_metrics(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> dict:
    """Dice, IoU, precision, recall for a batch of binary [B, 1, H, W]
    tensors, each computed per-sample then averaged across the batch (same
    convention the original compute_dice_iou used, extended with precision/
    recall via the same TP/FP/FN decomposition rather than a separate
    formula -- Dice and IoU below are algebraically identical to the
    original implementation, just derived from tp/fp/fn instead of
    intersection/union directly):

      TP = |P & G|, FP = |P & ~G|, FN = |~P & G|
      Dice      = (2*TP + eps) / (2*TP + FP + FN + eps)
      IoU       = (TP + eps)   / (TP + FP + FN + eps)
      Precision = (TP + eps)   / (TP + FP + eps)
      Recall    = (TP + eps)   / (TP + FN + eps)
    """
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


# --------------------------------------------------------------------------
# Hausdorff Distance (95th percentile) -- per-image only, not batch-vectorized
# --------------------------------------------------------------------------
def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """The outer ring of foreground pixels -- mask minus its own erosion.
    Standard Hausdorff distance is defined over object BOUNDARIES, not
    filled regions (using filled regions would let two blobs register as
    "close" via their interiors even if their actual edges disagree)."""
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff Distance between two 2D binary
    masks, in pixels (at whatever resolution the masks are given -- 256x256
    throughout this project). Using the 95th percentile rather than the raw
    maximum is standard practice in medical segmentation (e.g. nnU-Net,
    MedPy's hd95) specifically because the true maximum is an outlier
    magnet -- a single stray mispredicted pixel far from the true tissue
    would otherwise dominate the whole score.

    Returns NaN (not a large sentinel value like 999) if either mask is
    completely empty, since Hausdorff distance is genuinely undefined with
    no boundary to measure from -- a numeric sentinel would silently
    corrupt any mean computed over multiple patients, whereas NaN requires
    the caller to explicitly decide how to handle it (this module's own
    aggregation uses np.nanmean and reports how many patients were
    excluded, rather than hiding them)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not pred.any() or not gt.any():
        return float("nan")

    pred_boundary = _extract_boundary(pred)
    gt_boundary = _extract_boundary(gt)

    # distance_transform_edt(~mask) gives, at every pixel, the Euclidean
    # distance to the nearest True pixel of `mask`. Evaluating that at the
    # OTHER mask's boundary pixels gives exactly the directed point-to-set
    # distances the (symmetric) Hausdorff distance is defined over.
    dt_pred = distance_transform_edt(~pred)
    dt_gt = distance_transform_edt(~gt)

    d_gt_to_pred = dt_pred[gt_boundary]
    d_pred_to_gt = dt_gt[pred_boundary]

    return float(max(np.percentile(d_gt_to_pred, 95), np.percentile(d_pred_to_gt, 95)))


# --------------------------------------------------------------------------
# Final test-set evaluation -- one-time, per-patient, for thesis reporting
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_final_test_set(
    model,
    dataset_cls,
    splits_csv,
    device,
    transform,
    tissue_type: str = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Runs `model` (already loaded with its best checkpoint, already
    .to(device)) over the TEST split -- the one split never touched by
    Optuna's hyperparameter search or early stopping, so these numbers are
    free of the selection bias that would come from reporting the same
    validation-set numbers used to pick the winning trial/checkpoint.

    Returns one row per patient (patient_id, country, dice, iou, precision,
    recall, hd95) -- deliberately NOT just an aggregate mean, so the same
    output feeds both a country-stratified breakdown and a paired
    significance test against another model's own per-patient CSV
    (compare_models_significance.py), neither of which is possible from an
    aggregate-only summary.

    Runs one patient at a time (batch_size=1) rather than batching, since
    the test sets here are small (~31-33 patients) and this only runs once
    per study -- simplicity and guaranteed correct patient_id bookkeeping
    matter more here than throughput."""
    import inspect

    dataset_kwargs = {}
    if tissue_type is not None and "tissue_type" in inspect.signature(dataset_cls.__init__).parameters:
        dataset_kwargs["tissue_type"] = tissue_type

    test_dataset = dataset_cls(split="test", splits_csv=splits_csv, transform=transform, **dataset_kwargs)
    splits_df = pd.read_csv(splits_csv).set_index("patient_id")

    model.eval()
    records = []
    for idx in range(len(test_dataset)):
        image, mask = test_dataset[idx]
        patient_id = test_dataset.df.loc[idx, "patient_id"]

        logits = model(image.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > threshold).float()
        mask_batched = mask.unsqueeze(0).to(device)

        batch_metrics = compute_batch_metrics(pred, mask_batched)

        pred_np = pred.squeeze().cpu().numpy().astype(bool)
        gt_np = mask.squeeze().cpu().numpy().astype(bool)
        hd95 = compute_hd95(pred_np, gt_np)

        records.append(
            {
                "patient_id": patient_id,
                "country": splits_df.loc[patient_id, "country"],
                "dice": batch_metrics["dice"],
                "iou": batch_metrics["iou"],
                "precision": batch_metrics["precision"],
                "recall": batch_metrics["recall"],
                "hd95": hd95,
            }
        )

    return pd.DataFrame(records)
