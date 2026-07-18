# Tech Stack & Development Rules — classification/ (Phase 4)

## Core stack
Same installed environment as the root project (same `venv/`, same PyTorch/torchvision/CUDA setup) — no new dependencies introduced. Uses `torchvision.models` pretrained weights (ResNet18, MobileNetV3-Small, EfficientNet-B0 — all confirmed available in the installed `torchvision 0.28.0`), `optuna`, `albumentations`, `scikit-learn` (`sklearn.metrics` for accuracy/precision/recall/F1/AUC, and `train_test_split` for the stratified split).

## Directory structure
```
classification/
  .gitignore              -- excludes data/raw/, data/processed/images/, outputs/checkpoints/
                              (own file, does NOT touch the root .gitignore)
  .project_memory/        -- this module's working-memory system (roadmap/status/rules)
  data/
    raw/                  -- full extraction of archive.zip (gitignored, regenerable)
    processed/
      images/palpebral/, images/forniceal_palpebral/  -- gitignored, regenerable
      metadata.csv, splits.csv, extraction_log.csv     -- tracked, small
  scripts/
    prepare_dataset.py    -- fresh, independent data extraction/labeling/splitting
    dataset.py             -- PyTorch Dataset + transforms
    trainer_engine.py      -- shared Optuna training engine
    train_{arch}_{tissue}.py  -- 6 thin entry points
  outputs/
    checkpoints/           -- gitignored
    logs/                   -- tracked (per-trial CSV + study summary JSON)
```

## Isolation rules (why this module exists as its own thing)
1. **Zero code dependency on the root project's Phase 0-3 pipeline.** Nothing under `classification/` imports from `scripts/`, `models/`, or reads from `data/processed/` at the root. The only thing shared with the root project is the immutable source archive, `archive.zip` — read fresh, never the root's already-processed output.
2. **This was a deliberate, explicit trade-off, not a default.** Reusing the root's already-fixed extraction utilities (iCCP repair, EXIF transpose, ELIMINATO scan, comma-decimal Hgb parsing) would have been less work and lower-risk, but the project author explicitly chose full reimplementation for genuine isolation. Because of this, any future data-quality bug found in one pipeline (root Phase 0 or this one) does **not** automatically get fixed in the other — they must be checked/fixed independently. Don't assume a root-project fix (or a `classification/` fix) has propagated to the other side.
3. **Never edit root project files from within this module's work**, and never edit this module's files while doing root-project (segmentation) work — keep the two changesets separable in git history.
4. **Own `.gitignore`, not the root one.** `classification/.gitignore` handles this module's large binaries (raw archive extraction, processed images, checkpoints) so a `git add classification/` never risks pulling in large binaries, without ever touching the root `.gitignore`.

## Development rules carried over from the root project (still apply here)
1. **Verify empirically, don't assume** — every claim in `02_current_status.md` is backed by an actual measurement or run, not inference (e.g. the `.convert("RGB")` background-color finding, the typo-tolerant crop-matching bug, the cross-check against root Phase 0's numbers).
2. **Never commit large binaries** — enforced by `classification/.gitignore`.
3. **Distinct naming per (architecture, tissue_type) combination** — every `trainer_engine.py` output is keyed off `model_name` (e.g. `resnet18_palpebral`, not `resnet18`), so no run silently overwrites another's checkpoint/logs.
4. **Don't execute expensive or consequential operations without an explicit go-ahead** — code gets written and structurally verified (import check, dry forward pass) first; real Optuna training runs wait for a separate, explicit confirmation each time.
5. **When a decision could quietly produce wrong results, ask rather than guess** — this is exactly how the Hgb-threshold question and the reuse-vs-reimplement question got resolved before any code was written (see `02_current_status.md`).
6. **Git discipline:** commit only when explicitly asked (asked, this session); push only when explicitly asked (not yet asked — do not push without a separate, explicit request).
