# Tech Stack & Development Rules — classification/ (Phase 4)

## Core stack
Same installed environment as the root project (same `venv/`, same PyTorch/torchvision/CUDA setup) — no new dependencies introduced. Uses `torchvision.models` pretrained weights, `optuna`, `albumentations`, `scikit-learn` (`sklearn.metrics` for accuracy/precision/recall/F1/AUC, and `train_test_split` for the stratified split).

**Architecture roster (9, as of the v2 expansion, 2026-07-28)** — all confirmed available in the installed `torchvision 0.28.0` with ImageNet-pretrained weights, all frozen-backbone + trained head: ResNet18, MobileNetV3-Small, EfficientNet-B0 (original 3), DenseNet121, ConvNeXt-Tiny, RegNetY-400MF (new CNNs), Swin-Tiny, ViT-B/16, ViT-L/16 (new transformers, lightweight/medium/heavy). Full rationale in `02_current_status.md`. Deliberately kept torchvision-native (no `timm`) to preserve the "no new dependencies" rule below.

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
    dataset.py             -- PyTorch Dataset + transforms, resolution-aware since v2
    trainer_engine.py      -- shared Optuna training engine, 9-architecture registry since v2
    train_{arch}_{tissue}.py  -- ORIGINAL 6 entry points (v1 protocol). RETIRED as of the
                              v2 expansion -- trainer_engine.py changed underneath them, so
                              re-running one as-is now executes the v2 protocol but would
                              overwrite the v1 checkpoint/logs under the old model_name. Do
                              not run these; kept on disk only as the v1 code record.
  v2_scripts/
    train_{arch}_{tissue}_v2.py  -- 18 thin entry points (9 architectures x 2 tissue types),
                              v2 protocol (100-epoch ceiling, patience=7, dropout_rate tuned).
                              Isolated in its own directory specifically so the expanded
                              experiments never get confused with the original 6 at a glance.
                              Imports trainer_engine.py from the sibling scripts/ directory
                              via an explicit `parent.parent / "scripts"` path (verified
                              working, since it's one level further away than the original
                              6 scripts' same-directory import).
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
3. **Distinct naming per (architecture, tissue_type) combination — extended to protocol versions too.** Every `trainer_engine.py` output is keyed off `model_name` (e.g. `resnet18_palpebral`, not `resnet18`), so no run silently overwrites another's checkpoint/logs. This was tested for real during the v2 expansion: changing `trainer_engine.py` (shared by every entry-point script) meant the original 6 scripts would have silently produced v2-protocol results under v1-named files if re-run — resolved by suffixing every v2 `model_name` with `_v2` and isolating the new entry points in `v2_scripts/`, never by editing the old scripts' names retroactively.
4. **Don't execute expensive or consequential operations without an explicit go-ahead** — code gets written and structurally verified (import check, dry forward pass) first; real Optuna training runs wait for a separate, explicit confirmation each time.
5. **When a decision could quietly produce wrong results, ask rather than guess** — this is exactly how the Hgb-threshold question and the reuse-vs-reimplement question got resolved before any code was written (see `02_current_status.md`), and how the v2 expansion's tissue-scope/dependency/compute-budget questions were resolved via explicit confirmation before implementation.
6. **Git discipline:** commit only when explicitly asked (asked, this session); push only when explicitly asked (not yet asked — do not push without a separate, explicit request).
7. **Update `.project_memory/` in lockstep with implementation, not after the fact** (explicit project-author instruction, 2026-07-28) — every implementation step during the v2 expansion (and beyond) gets its corresponding `.project_memory` update in the same turn, not batched up for later. Scope stays limited to `classification/.project_memory/` per isolation rule #3 above; the root `CLAUDE.md`/`.project_memory/` are a separate, not-touched-from-here system.
8. **Never track or push private/local-only material — a repo-wide rule, not classification-specific.** `Source/` (literature PDFs, research summaries) is now `.gitignore`'d at the repo root, and the standing rule lives in root `CLAUDE.md` (added 2026-07-28, project-author-directed, full detail there). This is a deliberate, explicit exception to isolation rule #3 above (root files aren't normally touched from classification work) — the project author directly asked for a repo-wide security rule, which is outside this module's own scope by nature. Isolation rule #3 is about not *casually* mixing the two changesets during ordinary classification work, not an absolute bar when the project author explicitly requests a root-level change.
