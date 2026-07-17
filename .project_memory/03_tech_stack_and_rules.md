# Tech Stack & Development Rules — EYES-DEFY-ANEMIA

## Core stack
- Python 3.14.6 (local `venv/`); PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, torchaudio 2.11.0+cu130 (GPU: local RTX 4050; Kaggle: T4×2).
- `albumentations` 2.0.8 — synchronized image+mask augmentation (nearest-neighbor interpolation for masks, so they stay binary).
- `opencv-python` 5.0.0 (`cv2`, GUI-enabled — swapped from `opencv-python-headless` for `cv2.selectROI` support) — SIFT/ORB feature matching, RANSAC homography, connected components.
- `optuna` 4.9.0 — TPE-sampler hyperparameter search.
- `pandas`, `numpy`, `Pillow`, `scikit-learn` (stratified splitting), `openpyxl`.

## Directory structure
```
scripts/              -- all pipeline code (phase0, dataset.py, build_aligned_dataset.py,
                          trainer_engine.py, train_*.py)
models/segmentation/  -- unet.py, attention_unet.py, resunet.py
data/processed/
  images/, masks/      -- crop-based Phase 0 output (gitignored, regenerable from archive.zip)
  aligned_raw/         -- images/masks gitignored; alignment_log.csv tracked
  metadata.csv, dataset_splits.csv  -- tracked
outputs/
  checkpoints/         -- trained weights (.pth, gitignored)
  logs/                -- per-trial CSV + best-trial JSON summaries (tracked)
CLAUDE.md              -- authoritative, thesis-grade methodology reference
.project_memory/       -- this lightweight working-memory system (roadmap/status/rules)
```

## Development rules we've established
1. **Verify empirically, don't assume.** Every "it works" claim in this project has been backed by an actual run plus concrete evidence (shape checks, pixel-count math, visual overlays) — e.g. the raw-photo alignment was confirmed via a geometric pixel-count ratio check *and* visual inspection across both countries before being trusted.
2. **Never commit large binaries.** Images, masks, zips, and model checkpoints are `.gitignore`'d; only code and small CSV/JSON metadata/logs are tracked.
3. **Distinct naming per (model, dataset) combination.** Every `trainer_engine.py` output is keyed off `model_name` — always use a name that won't silently overwrite an existing result (e.g. `unet_aligned`, not `unet`).
4. **`CLAUDE.md` is the authoritative, cite-able record.** It must stay scientifically accurate. Anything not independently verified in this session (e.g. externally-reported Kaggle numbers) is explicitly labeled as such, never stated as confirmed fact.
5. **Don't execute expensive or consequential operations without an explicit go-ahead.** Code gets written and structurally verified (syntax check, safe import check, small synthetic tests) first; real training runs, git pushes, and other hard-to-reverse actions wait for explicit confirmation, given each time (not assumed from a prior approval).
6. **Reuse, don't duplicate.** Shared logic (e.g. `find_source_files`, `pad_to_square`, transform builders) is imported from its original module rather than copy-pasted.
7. **When a decision could quietly produce wrong results, ask rather than guess** (e.g. which image should pair with which mask; how to handle a missing `archive.zip`; how to log a disputed/unverifiable metric).
8. **Git discipline:** commit only when explicitly asked; push only when explicitly asked (a separate ask from commit); review `git status` before staging to catch stray/unexpected files before they're committed.
9. **Plan before executing non-trivial code changes** — think through the design (and, when a decision is ambiguous or consequential, say so or ask) before writing/running it.
