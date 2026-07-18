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
- [ ] Structural verification (import check + one dry forward pass per architecture/tissue combo) — in progress
- [ ] Actual Optuna training runs (all 6) — not started, separate explicit go-ahead needed before consuming real GPU time

## Not yet started
- [ ] Compare palpebral vs. forniceal_palpebral as classification input across all 3 architectures
- [ ] Decide on a winning (architecture, tissue_type) combination
- [ ] Any downstream integration with the root project (e.g. whether a winning classifier eventually gets referenced from the thesis alongside the segmentation results) — out of scope for now, deliberately not decided yet
