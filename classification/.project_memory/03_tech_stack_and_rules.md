# Tech Stack & Development Rules — classification/ (Phase 4)

## Core stack
Same installed environment as the root project (same `venv/`, same PyTorch/torchvision/CUDA setup) — no new dependencies introduced. Uses `torchvision.models` pretrained weights, `optuna`, `albumentations`, `scikit-learn` (`sklearn.metrics` for accuracy/precision/recall/F1/AUC, and `train_test_split` for the stratified split).

**Architecture roster (9, as of the v2 expansion, 2026-07-28)** — all confirmed available in the installed `torchvision 0.28.0` with ImageNet-pretrained weights, all frozen-backbone + trained head: ResNet18, MobileNetV3-Small, EfficientNet-B0 (original 3), DenseNet121, ConvNeXt-Tiny, RegNetY-400MF (new CNNs), Swin-Tiny, ViT-B/16, ViT-L/16 (new transformers, lightweight/medium/heavy). Full rationale in `02_current_status.md`. Deliberately kept torchvision-native (no `timm`) to preserve the "no new dependencies" rule below.

## Directory structure
**Renamed 2026-07-31:** `scripts/` -> `datapreparepipeline/` (git-mv'd, history preserved). The original 6 v1 entry-point scripts (`train_{arch}_{tissue}.py`, already retired/do-not-run as of the v2 expansion) were deleted outright rather than carried over, since nothing still depended on them. Every import/path reference to the old `scripts/` name (18 `v2_scripts/*.py` entry points, the CV pipeline's `cv_dataset.py`/`cv_trainer_engine.py`, `dataset.py`'s own docstring, `.gitignore`, and the 3 Kaggle notebooks) was updated and re-verified (`py_compile` + a real runtime import) to resolve against the new name. See `02_current_status.md` for the full account.
```
classification/
  .gitignore              -- excludes data/raw/, data/processed/images/, outputs/checkpoints/
                              (own file, does NOT touch the root .gitignore)
  .project_memory/        -- this module's working-memory system, one file per phase (rule #9 below)
    01_roadmap.md          -- checklist, all phases
    02_current_status.md   -- lean current-state snapshot only, as of the phase-file split (2026-08-02)
    03_tech_stack_and_rules.md  -- this file
    04_literature_review_findings.md  -- threshold/bias literature review (2026-07-28)
    05_kaggle_training_phase.md  -- clean-data 18-combo Kaggle sweep (2026-08-02 on)
    kaggle/01_kaggle_notes.md    -- Kaggle execution environment notes (paths, quirks, recipes)
  data/
    raw/                  -- full extraction of archive.zip (gitignored, regenerable)
    processed/
      images/palpebral/, images/forniceal_palpebral/  -- gitignored, regenerable
      metadata.csv, splits.csv, extraction_log.csv     -- tracked, small
  datapreparepipeline/    -- (renamed from scripts/, 2026-07-31)
    prepare_dataset.py    -- fresh, independent data extraction/labeling/splitting
    dataset.py             -- PyTorch Dataset + transforms, resolution-aware since v2
    trainer_engine.py      -- shared Optuna training engine, 9-architecture registry since v2
    efficientnet_b0_forniceal_5fold_cv/  -- dedicated 5-fold CV pipeline (cv_dataset.py,
                              cv_trainer_engine.py, run_cv_training.py, its own outputs/)
    (the original 6 v1 entry-point scripts, train_{arch}_{tissue}.py, were deleted
     2026-07-31 -- they were already retired/do-not-run as of the v2 expansion below)
  v2_scripts/
    train_{arch}_{tissue}_v2.py  -- 18 thin entry points (9 architectures x 2 tissue types),
                              v2 protocol (100-epoch ceiling, patience=7, dropout_rate tuned).
                              Isolated in its own directory specifically so the expanded
                              experiments never get confused with the original 6 at a glance.
                              Imports trainer_engine.py from the sibling datapreparepipeline/
                              directory via an explicit `parent.parent / "datapreparepipeline"`
                              path (verified working, since it's one level further away than
                              the original 6 scripts' same-directory import).
  outputs/
    checkpoints/           -- gitignored
    logs/                   -- tracked (per-trial CSV + study summary JSON)
```

## Isolation rules (why this module exists as its own thing)
1. **Zero code dependency on the root project's Phase 0-3 pipeline.** Nothing under `classification/` imports from `Segmentation/scripts/`, `Segmentation/models/`, or reads from `Segmentation/data/processed/` (paths as of the 2026-07-29 repo reorganization that moved the segmentation phase into its own top-level `Segmentation/` folder, sibling to `classification/`). The only thing shared with the root project is the immutable source archive, `archive.zip` — read fresh, never the root's already-processed output.
2. **This was a deliberate, explicit trade-off, not a default.** Reusing the root's already-fixed extraction utilities (iCCP repair, EXIF transpose, ELIMINATO scan, comma-decimal Hgb parsing) would have been less work and lower-risk, but the project author explicitly chose full reimplementation for genuine isolation. Because of this, any future data-quality bug found in one pipeline (root Phase 0 or this one) does **not** automatically get fixed in the other — they must be checked/fixed independently. Don't assume a root-project fix (or a `classification/` fix) has propagated to the other side.
3. **Never edit root project files from within this module's work**, and never edit this module's files while doing root-project (segmentation) work — keep the two changesets separable in git history.
4. **Own `.gitignore`, not the root one.** `classification/.gitignore` handles this module's large binaries (raw archive extraction, processed images, checkpoints) so a `git add classification/` never risks pulling in large binaries, without ever touching the root `.gitignore`.

## Development rules carried over from the root project (still apply here)
1. **Verify empirically, don't assume** — every claim in `02_current_status.md` is backed by an actual measurement or run, not inference (e.g. the `.convert("RGB")` background-color finding, the typo-tolerant crop-matching bug, the cross-check against root Phase 0's numbers).
2. **Never commit large binaries** — enforced by `classification/.gitignore`.
3. **Distinct naming per (architecture, tissue_type) combination — extended to protocol versions too.** Every `trainer_engine.py` output is keyed off `model_name` (e.g. `resnet18_palpebral`, not `resnet18`), so no run silently overwrites another's checkpoint/logs. This was tested for real during the v2 expansion: changing `trainer_engine.py` (shared by every entry-point script) meant the original 6 scripts would have silently produced v2-protocol results under v1-named files if re-run — resolved by suffixing every v2 `model_name` with `_v2` and isolating the new entry points in `v2_scripts/`, never by editing the old scripts' names retroactively.
4. **Don't execute expensive or consequential operations without an explicit go-ahead** — code gets written and structurally verified (import check, dry forward pass) first; real Optuna training runs wait for a separate, explicit confirmation each time.
5. **When a decision could quietly produce wrong results, ask rather than guess** — this is exactly how the Hgb-threshold question and the reuse-vs-reimplement question got resolved before any code was written (see `02_current_status.md`), and how the v2 expansion's tissue-scope/dependency/compute-budget questions were resolved via explicit confirmation before implementation.
6. **Git discipline:** commit only when explicitly asked; push only when explicitly asked. (Both have since been asked and done repeatedly, starting 2026-07-31 — this rule is about the *standing default* between requests, not a one-time gate that's now permanently open. Each commit/push still needs its own explicit ask.)
7. **Update `.project_memory/` in lockstep with implementation, not after the fact** (explicit project-author instruction, 2026-07-28) — every implementation step gets its corresponding `.project_memory` update in the same turn, not batched up for later. Scope stays limited to `classification/.project_memory/` per isolation rule #3 above; the root `CLAUDE.md`/`.project_memory/` are a separate, not-touched-from-here system.
8. **Never track or push private/local-only material — a repo-wide rule, not classification-specific.** `Source/` (literature PDFs, research summaries) is now `.gitignore`'d at the repo root, and the standing rule lives in root `CLAUDE.md` (added 2026-07-28, project-author-directed, full detail there). This is a deliberate, explicit exception to isolation rule #3 above (root files aren't normally touched from classification work) — the project author directly asked for a repo-wide security rule, which is outside this module's own scope by nature. Isolation rule #3 is about not *casually* mixing the two changesets during ordinary classification work, not an absolute bar when the project author explicitly requests a root-level change.
9. **One file per operational phase/major milestone, not perpetual append to `02_current_status.md`** (explicit project-author instruction, 2026-08-02, given `02_current_status.md` had grown large and dense enough to risk context overload). Going forward:
   - A new phase or major milestone (e.g. a training run, a new investigation, a pipeline overhaul) gets its own sequentially-numbered file — `05_kaggle_training_phase.md`, `06_...`, etc. — not another dated section appended to an existing file.
   - `02_current_status.md` is reserved for a **lean, current-state snapshot** — what's true right now and what's actively in flight — not a running chronological log. Superseded/historical detail belongs in the phase file it originated from, referenced by name rather than duplicated.
   - `01_roadmap.md` still tracks the checklist across all phases (it's a index of *done/pending*, not detailed narrative, so it doesn't have the same bloat problem) and continues to be updated in place.
   - Each new phase file should still open with the same "what is this, when did it start, what's it for" framing the existing numbered files use, so it's readable standalone without needing `02_current_status.md` as context.
   - When in doubt about whether something is "a new phase" vs. "more detail on the current one," default to a new file — splitting too early costs a cross-reference link; splitting too late is exactly the density problem this rule exists to prevent.
