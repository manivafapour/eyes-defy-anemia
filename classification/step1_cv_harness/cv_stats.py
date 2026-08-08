"""
Step 1 -- Measurement Harness: AUC estimation and bootstrap confidence
intervals.

The estimator, stated precisely so it can be defended in writing:

  * Within one repeat, every patient in the pool holds out exactly once, so
    concatenating the 5 folds' predictions yields one out-of-fold probability
    per patient. AUC is computed ONCE over that pooled vector, not as a mean
    of 5 per-fold AUCs. Mean-of-fold-AUCs would compute each country's AUC on
    ~1/5 of the pairs and then average the noise; pooling uses all 1,311
    India pairs at once.

  * Point estimate = mean of the R per-repeat pooled AUCs.

  * The CI must propagate TWO independent sources of variability: which
    patients happen to be in the cohort, and which model instance happened to
    be fit. So each bootstrap replicate draws a repeat uniformly at random
    AND resamples patients (stratified, with replacement, within each
    country x label cell so cell sizes are preserved exactly). A CI from
    patient resampling alone would be too narrow -- it would silently claim
    the estimate is more stable than repeated refitting shows it to be.

  * The India/Italy gap is computed INSIDE each replicate (same repeat, same
    resampled patients) so its CI accounts for the correlation between the
    two country AUCs rather than treating them as independent.

Also provides paired_delta_auc(), unused by Step 1 itself but part of its
contract: Steps 3-5 compare an intervention against this frozen baseline on
identical fold assignments, and a paired bootstrap on the DIFFERENCE has far
lower variance than comparing two independent CIs. With 23 India-healthy
patients, the unpaired comparison would be unable to detect any realistic
effect; the paired one is the only version with a chance of clearing.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from cv_config import CI_ALPHA, COUNTRIES, N_BOOTSTRAP, SEED


def safe_auc(labels: np.ndarray, probs: np.ndarray):
    """None (not 0.5, and not a crash) when a slice contains a single class --
    a bootstrap replicate can legitimately draw an all-anemic India-healthy
    resample, and silently coercing that to a number would bias the CI."""
    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, np.asarray(probs)))


def fold_level_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Full metric set for ONE fold's own held-out predictions (~37 patients) --
    confusion matrix, ROC curve, and the same accuracy/precision/recall/
    specificity/balanced_accuracy/f1/auc set datapreparepipeline/
    trainer_engine.py's compute_metrics() reports elsewhere in this project,
    kept field-for-field comparable rather than inventing a parallel schema.

    This is a per-fold diagnostic (feeds the loss-curve/confusion-matrix/ROC
    grids in cv_plots.py), NOT the pooled statistic bootstrap_auc_cis()
    computes above -- a single fold's ~37 patients is exactly the small-N
    regime this whole harness exists to move away from as the headline
    number, so these values are for visual per-fold inspection, not for
    citing as "the" result.

    None (not 0.0) for specificity/auc/roc_curve when undefined on this
    fold's slice (e.g. a fold with zero true negatives, or only one class
    present) -- never silently misread as a real zero.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n = len(labels)
    if n == 0:
        return {"n": 0}
    preds = (probs > threshold).astype(float)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = float(recall_score(labels, preds, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    out = {
        "n": int(n),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2) if specificity is not None else None,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    auc = safe_auc(labels, probs)
    out["auc"] = auc
    if auc is not None:
        fpr, tpr, _thresholds = roc_curve(labels, probs)
        out["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    else:
        out["roc_curve"] = None
    return out


def auc_by_country(labels: np.ndarray, probs: np.ndarray, countries: np.ndarray) -> dict:
    out = {"overall": safe_auc(labels, probs)}
    for country in COUNTRIES:
        mask = countries == country
        out[country] = safe_auc(labels[mask], probs[mask])
    india, italy = out.get("India"), out.get("Italy")
    out["india_italy_gap"] = None if (india is None or italy is None) else float(india - italy)
    return out


def _cell_indices(countries: np.ndarray, labels: np.ndarray) -> list:
    """Index groups for the 4 country x label cells. Resampling within these
    preserves the pool's exact composition (57/23 India, 20/84 Italy), which
    is what makes the replicate AUCs comparable to the observed one."""
    groups = []
    for country in COUNTRIES:
        for label_value in (0.0, 1.0):
            idx = np.where((countries == country) & (labels == label_value))[0]
            if idx.size:
                groups.append(idx)
    return groups


def bootstrap_auc_cis(
    per_repeat: list,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = CI_ALPHA,
    seed: int = SEED,
) -> dict:
    """per_repeat: list (one entry per repeat) of dicts with aligned 1D arrays
    'labels', 'probs', 'countries' -- the pooled out-of-fold predictions for
    that repeat, in a patient order that is identical across repeats.

    Returns point estimates, percentile CIs, and the per-repeat spread for
    'overall', 'India', 'Italy' and 'india_italy_gap'.
    """
    rng = np.random.default_rng(seed)
    n_repeats = len(per_repeat)
    if n_repeats == 0:
        raise ValueError("per_repeat is empty -- no pooled out-of-fold predictions to summarize.")

    labels = per_repeat[0]["labels"]
    countries = per_repeat[0]["countries"]
    for r in per_repeat[1:]:
        if not np.array_equal(r["labels"], labels) or not np.array_equal(r["countries"], countries):
            raise ValueError(
                "Repeats are not patient-aligned. Every repeat must hold out the same "
                "patient set in the same order, or resampled indices would not refer "
                "to the same patients across repeats."
            )

    observed = [auc_by_country(labels, r["probs"], countries) for r in per_repeat]
    keys = ["overall", "India", "Italy", "india_italy_gap"]
    point = {}
    spread = {}
    for key in keys:
        vals = [o[key] for o in observed if o[key] is not None]
        point[key] = float(np.mean(vals)) if vals else None
        spread[key] = {
            "per_repeat": [None if o[key] is None else float(o[key]) for o in observed],
            "sd_across_repeats": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        }

    groups = _cell_indices(countries, labels)
    replicates = {key: [] for key in keys}
    for _ in range(n_bootstrap):
        repeat_idx = rng.integers(0, n_repeats)                      # model-refit variability
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])  # patient variability
        rep = auc_by_country(labels[idx], per_repeat[repeat_idx]["probs"][idx], countries[idx])
        for key in keys:
            if rep[key] is not None:
                replicates[key].append(rep[key])

    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    result = {}
    for key in keys:
        vals = np.array(replicates[key], dtype=float)
        if vals.size == 0:
            result[key] = {"point": point[key], "ci_low": None, "ci_high": None, "ci_half_width": None}
        else:
            lo, hi = float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q))
            result[key] = {
                "point": point[key],
                "ci_low": lo,
                "ci_high": hi,
                "ci_half_width": float((hi - lo) / 2),
                "n_valid_replicates": int(vals.size),
            }
        result[key].update(spread[key])
    return result


def paired_delta_auc(
    baseline_per_repeat: list,
    intervention_per_repeat: list,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = CI_ALPHA,
    seed: int = SEED,
    pair_repeats: bool = True,
) -> dict:
    """Paired bootstrap on (intervention - baseline) AUC, for Steps 3-5.

    The pairing is what makes small effects detectable: both arms must have
    been run on the SAME fold assignments and the same patients, so each
    replicate resamples one patient index set and applies it to both arms,
    cancelling the cohort-composition variance that dominates the unpaired
    comparison. An effect is credible when the resulting CI on the difference
    excludes 0 -- not when the two arms' individual CIs fail to overlap.

    pair_repeats (default True) additionally pairs the repeat index, which is
    valid ONLY because both arms run on the same persisted fold assignments:
    repeat r of each arm sees identical fold geometry, so differencing within
    r cancels fold-composition variance too, leaving just the two arms' own
    initialization/augmentation noise. Set False for a deliberately
    conservative comparison that draws each arm's repeat independently --
    wider intervals, and the only correct choice if the two arms were ever run
    with different fold assignments.
    """
    rng = np.random.default_rng(seed)
    base0, interv0 = baseline_per_repeat[0], intervention_per_repeat[0]
    labels = base0["labels"]
    countries = base0["countries"]

    # Two distinct ways the pairing can be invalid, reported distinctly -- a
    # single merged message sent an earlier version of this check chasing the
    # wrong cause.
    if "patient_ids" in base0 and "patient_ids" in interv0:
        if not np.array_equal(base0["patient_ids"], interv0["patient_ids"]):
            raise ValueError(
                "Baseline and intervention cover different patients (or the same patients in a "
                "different order). A paired bootstrap requires an identical, identically-ordered "
                "patient set -- re-run the intervention against the baseline's fold_manifest.json."
            )
    if not np.array_equal(interv0["labels"], labels):
        raise ValueError(
            "Baseline and intervention disagree on the ground-truth labels. This is expected if one "
            "arm is a label-shuffle negative control -- controls are validated against chance "
            "(aggregate_baseline.py gate 5), not paired against the real arm."
        )
    if not np.array_equal(interv0["countries"], countries):
        raise ValueError("Baseline and intervention disagree on patient country assignments.")

    groups = _cell_indices(countries, labels)
    keys = ["overall", "India", "Italy", "india_italy_gap"]
    replicates = {key: [] for key in keys}
    n_shared = min(len(baseline_per_repeat), len(intervention_per_repeat))
    for _ in range(n_bootstrap):
        if pair_repeats:
            b_idx = i_idx = rng.integers(0, n_shared)
        else:
            b_idx = rng.integers(0, len(baseline_per_repeat))
            i_idx = rng.integers(0, len(intervention_per_repeat))
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        base = auc_by_country(labels[idx], baseline_per_repeat[b_idx]["probs"][idx], countries[idx])
        interv = auc_by_country(labels[idx], intervention_per_repeat[i_idx]["probs"][idx], countries[idx])
        for key in keys:
            if base[key] is not None and interv[key] is not None:
                replicates[key].append(interv[key] - base[key])

    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    out = {}
    for key in keys:
        vals = np.array(replicates[key], dtype=float)
        if vals.size == 0:
            out[key] = {"delta": None, "ci_low": None, "ci_high": None, "excludes_zero": None}
            continue
        lo, hi = float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q))
        out[key] = {
            "delta": float(np.mean(vals)),
            "ci_low": lo,
            "ci_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_valid_replicates": int(vals.size),
        }
    return out
