# Tech Stack & Development Rules — classification/ (Phase 4)

## Core stack
Same installed environment as the root project (same `venv/`, same PyTorch/torchvision/CUDA setup). Uses `torchvision.models` pretrained weights, `optuna`, `albumentations`, `scikit-learn` (`sklearn.metrics` for accuracy/precision/recall/F1/AUC, and `train_test_split` for the stratified split).

**Architecture roster (9, as of the v2 expansion, 2026-07-28)** — all confirmed available in the installed `torchvision 0.28.0` with ImageNet-pretrained weights, all frozen-backbone + trained head: ResNet18, MobileNetV3-Small, EfficientNet-B0 (original 3), DenseNet121, ConvNeXt-Tiny, RegNetY-400MF (new CNNs), Swin-Tiny, ViT-B/16, ViT-L/16 (new transformers, lightweight/medium/heavy). Full rationale in `02_current_status.md`. Kept deliberately torchvision-native (no `timm`) at the time.

**`timm` adopted as a real dependency (2026-08-08), for the `new_way/` roster (built same day — see `08_new_way_architecture_roster.md`).** The previous "no new dependencies" stance held only as long as torchvision's model zoo covered what was needed — it doesn't for two specific architectures the project author wants: **MaxViT-Small** and **CoAtNet-3** are not in torchvision at all (verified directly: torchvision's *only* MaxViT variant is `maxvit_t`; CoAtNet was never ported to torchvision in any size). `timm==1.0.28` added to root `requirements.txt`, along with its two real transitive dependencies `huggingface_hub==1.26.0` and `safetensors==0.8.0` (it was already incidentally present in the venv, unpinned and untracked, before this — now formally adopted and recorded).

**Verified before calling it done, not just installed:** both target models built with `pretrained=True`, real weights downloaded from Hugging Face Hub, and a forward pass run — not just import-checked.
- `maxvit_small_tf_224.in1k` — **68.16M params**, output shape `(1, 768)`. Pretrained tag `in1k` — **standard ImageNet-1k weights, same regime as every torchvision model in this project.** No inconsistency here.
- `coatnet_3_rw_224.sw_in12k` — **163.64M params**, output shape `(1, 1536)`. Pretrained tag `sw_in12k` — trained on the larger **ImageNet-12k**, with **no ImageNet-1k fine-tuning stage** (confirmed: `timm.list_models('coatnet*', pretrained=True)` has no `coatnet_3_rw_224.*_ft_in1k` variant, unlike e.g. `coatnet_2_rw_224.sw_in12k_ft_in1k`). This is the one architecture in the planned roster with a genuinely different pretraining regime from everything else — worth an explicit note if reported in the thesis, not treated as just another interchangeable pretrained backbone.

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
    06_efficientnet_b0_5fold_cv_deep_dive.md  -- consolidated EfficientNet-B0/forniceal_palpebral
                              5-fold CV + Grad-CAM record (2026-07-30/08-01 work, file created 08-02).
                              HISTORICAL RECORD ONLY -- that pipeline is deprecated (see below)
    07_step1_measurement_harness.md  -- defensibility/interpretability programme, Step 1
                              (pooled out-of-fold repeated CV + bootstrap CIs, 2026-08-04 on)
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
    efficientnet_b0_forniceal_5fold_cv/  -- DEPRECATED (project author, 2026-08-04).
                              Dedicated 5-fold CV pipeline (cv_dataset.py, cv_trainer_engine.py,
                              run_cv_training.py, its own outputs/). Its results were produced on
                              pre-fix data (white-background bug), so they are not comparable with
                              any _v2_clean result; the author intends to delete the folder. DO NOT
                              extend, import from, or cite its numbers -- in particular the
                              "recall 1.000 / specificity 0.35 across 5 folds" finding is about
                              THIS pipeline only and was mistakenly attributed to the v2_clean
                              models once (2026-08-04). Superseded by step1_cv_harness/ below,
                              which was deliberately built as a clean slate rather than extending
                              it. Historical record kept in .project_memory/06_....md.
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
  v2_clean_scripts/
    train_{arch}_{tissue}_v2_clean.py  -- 18 entry points, same v2 protocol against the
                              white-background-bug-fixed data. Its outputs/ holds all 18
                              completed combos (CNN + transformer batches both done).
  step1_cv_harness/         -- NEW 2026-08-04. Defensibility programme, Step 1: pooled
                              out-of-fold repeated stratified CV + bootstrap CIs, replacing
                              the single split's 40-discordant-pair India AUC (CI half-width
                              ~+/-0.27) with a 1,311-pair estimate. cv_config.py, cv_data.py,
                              cv_engine.py, cv_stats.py, validate_harness.py,
                              run_cv_harness.py, aggregate_baseline.py, README.md, outputs/.
                              IMPORTS (never copies) the live datapreparepipeline/ architecture
                              builders and transforms, so the models it measures are
                              bit-identical to what v2_clean trained -- a forked copy would
                              silently drift and stop being a valid baseline. Writes NO
                              checkpoints by design (predictions, not weights).
  new_way/                  -- NEW 2026-08-08. Fresh 8-architecture roster (5 CNN + 3
                              Hybrid; ViT deliberately dropped -- vit_b_16/vit_l_16 already
                              fully trained in v2_clean), 16 entry points (8 archs x 2
                              tissue types), model_name suffix _new_way. Same shared
                              trainer_engine.py/ARCHITECTURE_REGISTRY mechanism as v2_clean
                              (extended, not a new engine) -- see 08_new_way_architecture_
                              roster.md. First user of the timm dependency (below).
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
10. **State the statistical power before reporting a comparison, and compare interventions with a PAIRED test** (added 2026-08-04, after finding that the India/Italy AUC gap column — the project's primary confound signal — rests on 40 discordant pairs, giving a 95% CI half-width of ~±0.27 that silently swallowed the entire observed 0.550–1.000 spread across 12 models).
    - Any per-country metric quoted from a single 33-patient split is a point estimate with a CI wider than most effects of interest. Report the CI or say the number is underpowered; never present such a ranking as a ranking of models.
    - A **systematic** effect across many independently-trained models (e.g. India < Italy in 11/12, sign test p=0.0064) can be solid even when every individual value is noise. Distinguish the two explicitly.
    - Interventions (Steps 3–5) must reuse the persisted fold assignments from `step1_cv_harness/outputs/{combo}/fold_manifest.json` and be compared via `cv_stats.paired_delta_auc()`, **not** by checking whether two independent CIs overlap. Measured on real out-of-fold predictions with a +0.03 probability shift: paired detected it (CI [+0.073, +0.252]), unpaired did not (CI [−0.043, +0.340]).

## Technical facts worth not rediscovering
- **`requires_grad=False` does NOT freeze BatchNorm running statistics.** `running_mean`/`running_var` keep updating on every forward pass in `train()` mode, which affects 5 of the 6 CNNs in this roster (all except the pure-transformer heads). Found 2026-08-04 while building `step1_cv_harness/cv_engine.py`: a best-epoch snapshot that captures only trainable parameters will restore the best head paired with the *last* epoch's normalization statistics. Snapshot buffers too. This also means "the backbone is frozen" is not strictly true for BN statistics in any run in this project, v2_clean included — worth stating precisely rather than loosely in the thesis.
- **With a frozen backbone and a `GAP → Dropout → Linear` head, Grad-CAM degenerates to exact CAM** — global average pooling makes the gradient w.r.t. each feature-map channel constant across space, so the CAM weights *are* the learned head weights. Exact rather than heuristic (a genuine defensibility argument), but it makes the method **cue-blind**: a colour/illumination shortcut is a channel reweighting, not a spatial displacement. Holds cleanly for ResNet18/EfficientNet-B0/RegNetY-400MF/DenseNet121; perturbed for ConvNeXt-Tiny (LayerNorm between pooling and linear) and MobileNetV3-Small (a frozen `Linear(576→1024)+Hardswish` sits in between).
