# Kaggle Training Phase — Clean-Data 18-Combo Sweep (classification/, Phase 4)

**Started:** 2026-08-02. First file under the new one-file-per-phase documentation convention (`03_tech_stack_and_rules.md` rule #9) — everything from here on about *this specific training phase* goes in this file, not appended to `02_current_status.md`.

## What this phase is

Retraining the full 18-combo v2 architecture sweep (9 architectures × 2 tissue types) against the dataset after the white-background bug fix, to establish bug-free baselines and find out whether/how much that bug was contributing to the India/Italy AUC gap and shortcut-learning behavior investigated over the prior sessions.

**Context this phase builds on** (full detail in `02_current_status.md`'s "READY FOR TRAINING" entry and earlier dated entries — not repeated here):
- White-background convention bug found and fixed in `prepare_dataset.py`; full dataset reprocessed; independently re-verified clean (428/428 images, 4 strict checks, zero failures).
- WHO labeling thresholds re-examined against Ramos-Soto et al. and reconfirmed (adopting their thresholds would widen the India/Italy gap, not narrow it).
- 18 new entry-point scripts (`classification/v2_clean_scripts/train_*_v2_clean.py`) and 2 Kaggle notebooks (`classification-cnn-clean.ipynb`, `classification-vit-clean.ipynb`) built, committed, and pushed.

## Status

**Not yet launched.** Waiting on:
1. Project author to upload the reprocessed `classification/data/processed/` as a new Kaggle dataset.
2. Both notebooks to be run on Kaggle (`classification-cnn-clean.ipynb` — 12 CNN combos; `classification-vit-clean.ipynb` — 6 transformer combos, `swin_t`+`vit_b_16`+`vit_l_16`).

No known blockers on the code or data side.

## Naming reference (for when results start coming in)

All 18 new results use the `_v2_clean` model-name suffix (e.g. `resnet18_palpebral_v2_clean`), distinct from both the original v1 and the dirty-data v2 results — all three generations stay on disk side by side, nothing gets silently overwritten.

## Log

### 2026-08-02 — Pre-launch config tuning: `n_startup_trials` and `batch_size`

Two changes to the shared engine before launch, both in code every one of the 18 `v2_clean_scripts` entry points inherits automatically (they call `run_study()` with no hyperparameter arguments of their own — all config lives in `trainer_engine.py`/`dataset.py`, not duplicated per script):

1. **`trainer_engine.py`: `TPESampler(seed=SEED, n_startup_trials=5)`**, was the Optuna default of 10. With `N_TRIALS=12`, the default meant 10/12 trials per combo were pure random sampling before TPE's Bayesian modeling ever engaged — verified directly against the installed Optuna version's actual default (`inspect.signature`, not assumed from general knowledge). 5 leaves a real 7-trial informed-search budget instead of 2. Applies identically to all 18 combos, not architecture-family-specific — a considered CNN-vs-Transformer LR-bounds split was discussed and deliberately deferred (see rationale below).
2. **`dataset.py`: `BATCH_SIZE` raised 16 → 32.** Safe because every backbone is frozen (only the head trains, 441–1,281 trainable params depending on architecture) — comfortable VRAM headroom on Kaggle's GPUs, halves optimizer steps/epoch across the sweep.

**Considered and deliberately NOT done: splitting the Optuna LR search bounds by architecture family** (narrower/lower range for the 3 transformers). Reasoning: the classic "ViT fine-tuning is unstable at high LR" risk is about training self-attention weights directly, which doesn't apply here since the backbone is frozen for every architecture — what's actually being tuned is a small linear head on frozen features, a much more benign optimization landscape. Splitting the bounds now, without pilot evidence, would also risk making Phase 1's architecture comparison unfair (Phase 2 selects top-2 CNN + top-2 transformer from these results) — a search-design asymmetry could get conflated with genuine architecture quality. Deferred to a possible evidence-based, narrower re-search scoped to just the Phase 2 champions before Phase 3's 3-fold CV, if Phase 1's own per-trial logs (`learning_rate` vs. `val_f1`) show real divergence patterns for transformers once results are in.

Verified before considering this done: both files `py_compile` clean; runtime check confirms `dataset.BATCH_SIZE` (32) propagates through `trainer_engine.py`'s import; a `v2_clean_scripts` entry point (`train_resnet18_palpebral_v2_clean.py`) confirmed to see `BATCH_SIZE=32` via the shared module, not a stale copy.

*(Further entries appended here as the training phase progresses — dataset upload confirmation, per-notebook run status, results extraction, analysis. Once this phase is complete and a new one starts, e.g. a results-analysis or mitigation-implementation phase, that gets its own `06_...` file in turn.)*
