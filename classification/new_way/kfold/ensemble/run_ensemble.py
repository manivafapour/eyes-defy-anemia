"""
Entry point: soft-voting ensembles over the 6 trained new_way/kfold/ combos
(Output/version2/checkpoints/), evaluated on the sealed test set. No
training happens here -- see ensemble_engine.py for the inference/merging
logic and classification/new_way/kfold/kfold_engine.py for how the
checkpoints being combined were produced.

Ranked by mean test F1 across their own 3 folds (Output/version2/logs/
*_kfold_summary.json): MaxViT-T (0.827) > RegNet-Y-16GF (0.776) ~
ConvNeXt-Base (0.775) > ConvNeXt-Large (0.763) > CoAtNet-3 (0.750) >
EfficientNet-B3 (0.650, trained on the forniceal_palpebral crop instead of
palpebral -- not a like-for-like comparison with the other five).

Ensembles tried here, cheapest/most-diverse first:
  - top2_best_fold / top3_best_fold / top5_palpebral_best_fold: each
    architecture contributes its single best-validation-F1 fold, ranging
    from the top 2 up to all 5 palpebral-tissue architectures.
  - top3_all_folds / top5_palpebral_all_folds: same architecture sets, but
    every fold of every architecture votes (3x / 5x more members).
  - all6_incl_efficientnet_best_fold: adds EfficientNet-B3 despite its
    weaker solo score and different input crop, to see whether a
    genuinely different input modality contributes useful diversity even
    from a weaker member.
  - maxvit_t_folds_only: the 3 folds of the single best architecture,
    voting against each other -- an in-architecture ensemble baseline, to
    separate "does averaging folds of the SAME model help" from "does
    combining DIFFERENT architectures help" (the latter is expected to
    matter more -- correlated errors within one architecture's 3 folds
    should cancel out less than errors across architecturally different
    models).
  - top2_palpebral_plus_efficientnet_best_fold: MaxViT-T + RegNet-Y-16GF +
    EfficientNet-B3, all best-fold. A more isolated test of whether the
    forniceal_palpebral crop contributes real cross-modality diversity --
    all6_incl_efficientnet_best_fold already showed adding EfficientNet-B3
    to the group of 5 hurt, but that result conflates two things:
    "does mixing tissue crops help" and "does adding your weakest overall
    model help" (EfficientNet-B3's own mean test F1, 0.650, trails every
    other architecture by a wide margin). Pairing it with just the top 2
    strong palpebral models instead removes most of the dilution-by-weak-
    members effect, so any change here is more attributable to the crop
    difference itself.

Soups (weight-averaging, not soft-voting -- see ensemble_engine.py's
"Model soups" section for why this is only valid WITHIN one architecture's
own folds, never across different architectures): for each of the 6
architectures, average its 3 fold checkpoints' weights into a single merged
model, then evaluate that one model -- one forward pass at inference, same
cost as any individual checkpoint, unlike an N-member soft-voting ensemble.
Directly comparable to each architecture's own all_folds() soft-voting
ensemble (e.g. maxvit_t_soup vs. maxvit_t_folds_only above), since both
combine the exact same 3 checkpoints -- the only thing that differs is
WHERE the averaging happens (weight-space before inference, vs.
probability-space after inference).

Stacking (via stacking_defs): a logistic-regression meta-learner fit on
out-of-fold (OOF) predictions from the given base architectures, applied to
their fold-averaged test-set predictions. Unlike greedy_soup, this IS
leak-free -- see ensemble_engine.py's "Stacking" section for why K-fold's
own structure makes OOF predictions free of the leakage problem that blocks
a clean greedy soup. Tries the same base-learner groupings as the
soft-voting ensembles above (top2/top3/top5/all6) so the two combination
strategies are directly comparable on identical inputs.

Greedy soup (per architecture, via greedy_soup_keys): the Wortsman et al.
2022 procedure -- try folds in descending order of their own individual
best_val_f1, keep each one only if it doesn't reduce the running soup's
score. IMPORTANT: see ensemble_engine.greedy_soup()'s docstring -- this
project's 3-fold split has no leak-free multi-fold validation set, so the
accept/reject decisions here are made against the sealed TEST set. That
makes the resulting "*_greedy_soup_TEST_SELECTED" rows optimistic
(selection leakage), unlike every other row in this comparison. Kept as
its own "kind" in the output and flagged in its own name for exactly that
reason -- read as "best subset this test set happened to reward", not as
an unbiased performance estimate.

Standalone-runnable: `python classification/new_way/kfold/ensemble/run_ensemble.py`
"""

import sys
from pathlib import Path

import pandas as pd

ENSEMBLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENSEMBLE_DIR))

from ensemble_engine import MODEL_REGISTRY, all_folds, best_fold, run_comparison  # noqa: E402

if __name__ == "__main__":
    ensemble_defs = {
        "top2_best_fold": best_fold("maxvit_t") + best_fold("regnet_y_16gf"),
        "top3_best_fold": best_fold("maxvit_t") + best_fold("regnet_y_16gf") + best_fold("convnext_base"),
        "top3_all_folds": all_folds("maxvit_t") + all_folds("regnet_y_16gf") + all_folds("convnext_base"),
        "top5_palpebral_best_fold": (
            best_fold("maxvit_t")
            + best_fold("regnet_y_16gf")
            + best_fold("convnext_base")
            + best_fold("convnext_large")
            + best_fold("coatnet_3")
        ),
        "top5_palpebral_all_folds": (
            all_folds("maxvit_t")
            + all_folds("regnet_y_16gf")
            + all_folds("convnext_base")
            + all_folds("convnext_large")
            + all_folds("coatnet_3")
        ),
        "all6_incl_efficientnet_best_fold": (
            best_fold("maxvit_t")
            + best_fold("regnet_y_16gf")
            + best_fold("convnext_base")
            + best_fold("convnext_large")
            + best_fold("coatnet_3")
            + best_fold("efficientnet_b3")
        ),
        "maxvit_t_folds_only": all_folds("maxvit_t"),
        "top2_palpebral_plus_efficientnet_best_fold": (
            best_fold("maxvit_t") + best_fold("regnet_y_16gf") + best_fold("efficientnet_b3")
        ),
    }

    soup_defs = {f"{model_key}_soup_all_folds": (model_key, [1, 2, 3]) for model_key in MODEL_REGISTRY}
    greedy_soup_keys = list(MODEL_REGISTRY)
    stacking_defs = {
        "stack_top2": ["maxvit_t", "regnet_y_16gf"],
        "stack_top3": ["maxvit_t", "regnet_y_16gf", "convnext_base"],
        "stack_top5_palpebral": ["maxvit_t", "regnet_y_16gf", "convnext_base", "convnext_large", "coatnet_3"],
        "stack_all6": list(MODEL_REGISTRY),
    }

    comparison_df = run_comparison(
        ensemble_defs, soup_defs=soup_defs, greedy_soup_keys=greedy_soup_keys, stacking_defs=stacking_defs
    )

    print("\n=== Full comparison (sorted by test F1) ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(comparison_df.sort_values("f1", ascending=False)[["name", "kind", "n_members", "f1", "accuracy", "auc"]].to_string(index=False))
