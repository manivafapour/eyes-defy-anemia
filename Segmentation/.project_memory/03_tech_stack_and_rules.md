# Tech Stack & Development Rules — EYES-DEFY-ANEMIA

## Core stack
- Python 3.14.6 (local `venv/`); PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, torchaudio 2.11.0+cu130 (GPU: local RTX 4050; Kaggle: T4×2).
- `albumentations` 2.0.8 — synchronized image+mask augmentation (nearest-neighbor interpolation for masks, so they stay binary).
- `opencv-python` 5.0.0 (`cv2`, GUI-enabled — swapped from `opencv-python-headless` for `cv2.selectROI` support) — SIFT/ORB feature matching, RANSAC homography, connected components.
- `optuna` 4.9.0 — TPE-sampler hyperparameter search.
- `pandas`, `numpy`, `Pillow`, `scikit-learn` (stratified splitting), `openpyxl`.

## Directory structure

**Moved into a dedicated `Segmentation/` top-level folder 2026-07-29** (project author request, for a clean, self-explanatory repo layout alongside the sibling `classification/` module). All paths below are relative to `Segmentation/`, which itself sits at the repo root next to `classification/`, `Source/`, and the shared `CLAUDE.md`/`requirements.txt`/`venv/`. No code changes were needed for this move — every script already computed its paths via `Path(__file__).resolve().parent[.parent]`, never `cwd`-relative or hardcoded, so the whole subtree moved as a self-consistent unit. `CLAUDE.md` deliberately stays at the repo root, not inside `Segmentation/` — see the note below.

```
D:\khaje\EYES-DEFY-ANEMIA\
  CLAUDE.md              -- stays at repo root (see note below), updated to reference
                             Segmentation/... paths throughout
  requirements.txt, venv/, .gitignore, README.md, LICENSE  -- shared, repo-wide, at root
  classification/        -- sibling module, untouched by this move
  Source/                -- private research material, untouched, gitignored
  Segmentation/
    scripts/              -- all pipeline code (phase0, dataset.py, build_aligned_dataset{,_forniceal}.py,
                              trainer_engine.py); train_pretrained/ -- 18 generated entry-point scripts
                              (9 architectures x 2 tissue types) + _generate_scripts.py
    models/segmentation/  -- pretrained_registry.py (9-model ARCHITECTURE_REGISTRY), transunet.py.
                              (The original unet.py/attention_unet.py/resunet.py -- 3 hand-built
                              models -- were removed 2026-08-08, superseded by the above; see
                              .project_memory/04_pretrained_architecture_sweep.md.)
    data/processed/
      images/               -- Phase 0 output (gitignored, regenerable from archive.zip); still used
                              by AnemiaClassificationDataset. (masks/, the crop-based segmentation
                              target, was removed alongside the 3 old models above.)
      aligned_raw/, aligned_raw_forniceal/  -- images/masks gitignored; alignment_log.csv tracked
      metadata.csv, dataset_splits.csv  -- tracked
    outputs/
      checkpoints/         -- trained weights (.pth, gitignored)
      logs/                -- per-trial CSV + best-trial JSON summaries (tracked)
    archive.zip            -- raw source data, gitignored
    .project_memory/       -- this lightweight working-memory system (roadmap/status/rules), split
                              into additional numbered topic files as they grow (see rule 11 below);
                              kaggle/01_kaggle_notes.md -- Kaggle execution specifics (mount paths,
                              environment quirks), split out same as classification/'s own kaggle/ folder
```

**Why `CLAUDE.md` stays at the repo root instead of moving into `Segmentation/`:** its own self-description is "the primary methodology reference for the project's written thesis" — a whole-project document (segmentation *and* classification), not segmentation-specific, even though its actual written content is 100% segmentation right now (classification/Phase 4 has never been written into it). Root placement also matters practically: Claude Code auto-loads a repo-root `CLAUDE.md` at the start of every session opened at `D:\khaje\EYES-DEFY-ANEMIA` — moving it into `Segmentation/` would silently stop that auto-loading for future sessions (the classification module's own `.project_memory/` already doesn't get auto-loaded, requiring explicit reference; the same would happen to this file if moved). `CLAUDE.md`'s own session-start instruction and every internal path reference were updated to point at `Segmentation/...` even though the file itself didn't move.

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
10. **Never track or push private/local-only material.** `Source/` (literature PDFs, research summaries, any other local-only documents) is `.gitignore`'d at the repo root — added 2026-07-28 (full rule in `CLAUDE.md`). Before any broad `git add`, check `git status` to confirm nothing under `Source/` or similar private directories is about to be staged, not just trusting `.gitignore`.
11. **Keep `.project_memory/` from growing unbounded — split into new numbered files by topic, not one ever-growing status file.** Added 2026-08-08 (project author's explicit request) after `02_current_status.md` accumulated multiple sessions' full narratives. When a topic is large/self-contained (e.g. a whole architecture-sweep effort), give it its own `0N_topic_name.md` file — mirrors `classification/.project_memory/`'s own numbered-file convention (which already went up to 07 plus a `kaggle/` subfolder). Cross-reference from `01_roadmap.md`/`02_current_status.md` with a one-line pointer rather than duplicating content across files. **Update memory files every time something is implemented OR removed** — not just additions; removals need the same "what, why, what broke, what didn't" record (see `04_pretrained_architecture_sweep.md`'s removal section for the template).
12. **When another Claude Code session may be working in this same repo concurrently, stage files by exact filename — never a directory glob (`git add Segmentation/`, `git add -A`).** Added 2026-08-08 after a real incident: a parallel session was concurrently committing unrelated `classification/new_way/` work while this session tried to commit 3 segmentation memory files. A broad `git add Segmentation/` silently produced "no changes to commit" because a race condition had already swept the staged files into the other session's commit under an unrelated message; that commit was then itself reset by the other process before this session could push it, leaving the files back in an untracked state. No data was lost — file content was verified intact on disk before each re-stage — but it took manual detection to catch, and would have been invisible with a broader `git add`. Explicit-filename staging plus a post-stage `git status` check keeps a commit's contents predictable even when another process is touching the index at the same time.
