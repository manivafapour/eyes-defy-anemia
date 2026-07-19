# Roadmap — classification/ (Phase 4: Anemia Classification)

High-level, sequential plan for this module only. `[x]` = done and verified (not just written). This directory is intentionally decoupled from the root project's Phase 0-3 segmentation work — see `03_tech_stack_and_rules.md` for the isolation rules.

## Data preparation
- [x] Extract `archive.zip` fresh into `data/raw/` (independent of the root project's `data/processed/`)
- [x] Fresh metadata parsing + WHO-threshold anemia labels (Male<13.0, Female<12.0 g/dL, same both countries) — NOT the country/gender-specific thresholds floated early in the session (India W<12.0/M<14.0, Italy flat<10.5); those were unsourced and would have amplified the India/Italy confound, so the project author explicitly chose WHO thresholds instead
- [x] 4-way (`country + anemic_label`) stratified 70/15/15 patient-level split
- [x] Extract both `palpebral` and `forniceal_palpebral` crops, flattened to a genuinely clean black background (verified: naive `.convert("RGB")` alone is NOT sufficient — see `02_current_status.md`)
- [x] Cross-validated against the root project's independently-computed Phase 0 numbers (217 patients, 126/91 label split, same per-country/split breakdown) — strong signal the fresh reimplementation is correct

## Model architecture & training infrastructure
- [x] Architecture registry: ResNet18, MobileNetV3-Small, EfficientNet-B0 — ImageNet-pretrained, frozen backbone, replaced single-logit head
- [x] Optuna-driven training engine (`trainer_engine.py`) — BCEWithLogitsLoss with `pos_weight`, metrics reported both aggregate and stratified by country
- [x] 6 thin entry-point scripts (3 architectures × 2 tissue types)
- [x] Structural verification (import check + one dry forward pass per architecture/tissue combo, all 6 combos)
- [x] End-to-end local dry-run (1 trial, 1 epoch, `train_resnet18_palpebral` config, run through the real `run_study()` path) — DataLoader, forward/backward, `BCEWithLogitsLoss`, and country-stratified metric computation/persistence all confirmed working with zero runtime errors. Dry-run artifacts (checkpoint/logs) deleted afterward, not committed.
- [x] Actual Optuna training runs (all 6) — **executed on Kaggle, results pulled back 2026-07-19.** All 6 completed the full 12-trial search. Results verified from `study_summary.json` for each + visual inspection of all plot types (not just the two spot-checked during local dry-run verification). See `02_current_status.md` for the full results table and analysis.

## Thesis-grade evaluation upgrade (implemented and verified 2026-07-18)
- [x] Stratified confusion matrix (overall + India + Italy), computed every epoch, plotted as a 3-panel figure for the best trial only
- [x] F1/precision/recall — already computed; now included directly in the per-country JSON summary alongside the new confusion matrix/ROC data
- [x] ROC curve plot (fpr/tpr persisted per epoch, plotted for the best trial's "overall" curve only; per-country fpr/tpr data is also computed and saved in the JSON even though only "overall" gets plotted, since only the confusion matrix was explicitly requested stratified)
- [x] Train vs. validation loss curves over epochs — per-epoch history now persisted via `trial.set_user_attr()` on every trial, plotted for the best trial only
- [x] Plotting strictly scoped to the single best trial per model (`study.best_trial`) — verified no plot is generated for non-winning trials
- [x] Verified via a real end-to-end local dry-run (1 trial, 1 epoch, resnet18/palpebral): all 3 plots (loss curve, ROC curve, confusion matrices) generated correctly, visually inspected, matched the JSON data exactly. Dry-run artifacts deleted after verification, not committed.
- Requested 2026-07-18, implemented and verified same day. See `02_current_status.md` for full detail.

## Kaggle results: comparison across all 6 combinations (2026-07-19)
- [x] Compare palpebral vs. forniceal_palpebral as classification input across all 3 architectures — done, see `02_current_status.md` for the full table
- [x] Identify best overall performer: **EfficientNet-B0 / forniceal_palpebral** (F1=Acc=AUC=0.903, only combo where all three headline metrics agree and clear the rest by a wide margin)
- [x] Identify best confound-handling: **MobileNetV3-Small / forniceal_palpebral** (smallest India/Italy AUC gap, 0.160; the only combo without a blanket recall=1.0, i.e. not just defaulting to "predict anemic")
- [x] **Systematic finding, not one model's quirk:** all 6 independently-tuned models show Italy AUC > India AUC, and 5 of 6 show recall=1.0 for the anemic class — consistent with the `pos_weight` loss term biasing toward "predict anemic," interacting differently with India's anemic-majority vs. Italy's non-anemic-majority val composition. ResNet18/forniceal_palpebral is the clearest case of likely confound exploitation: 78.6% India accuracy paired with a sub-chance 0.450 India AUC.
- [ ] **Not yet decided:** which single (architecture, tissue_type) to carry forward as "the" Phase 4 result, if the thesis needs one — current recommendation is to report both winners (best-overall vs. best-confound-handling) rather than collapsing to one, since they disagree.
- [x] Moved the real Kaggle logs/plots from the doubly-nested `classification/classification/outputs/` into the canonical `classification/outputs/{logs,plots}/` location, verified integrity (all JSONs parse, all 18 PNGs pass `Image.verify()`), and committed as the official experimental record. Checkpoints moved to `classification/outputs/checkpoints/` too but stay local-only (gitignored) per explicit instruction. Also added `*.zip` to `classification/.gitignore` after finding a 125MB leftover results zip with no ignore coverage.
- [ ] Any downstream integration with the root project (e.g. whether a winning classifier eventually gets referenced from the thesis alongside the segmentation results) — out of scope for now, deliberately not decided yet
