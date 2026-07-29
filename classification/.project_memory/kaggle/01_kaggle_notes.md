# Kaggle Execution Notes — classification/ (Phase 4)

Dedicated log for everything specific to running this module's training on Kaggle (paths, environment quirks, workflow patterns) — kept separate from the main `.project_memory` files, which cover the pipeline/architecture/results side, not the execution environment. Created 2026-07-28 at the project author's request, after the dataset-mount-path issue below cost real debugging time and was judged worth recording so it doesn't happen again.

## Critical finding: Kaggle's actual input mount path does not match the dataset's display name

**What was assumed (wrong):** a Kaggle dataset named `Processed-dataset` (uploaded from `D:\khaje\EYES-DEFY-ANEMIA\classification\data\processed\dataset.zip`) would mount at `/kaggle/input/processed-dataset/` (Kaggle lowercases/kebab-cases display names into slugs) or possibly `/kaggle/input/Processed-dataset/`.

**What actually happened:** neither. The real, working path was:
```
/kaggle/input/datasets/manivafapour33/processed-dataset
```
i.e. nested two levels deeper than the typical `/kaggle/input/<slug>/` pattern, under `datasets/<kaggle-username>/<slug>/`. Confirmed working empirically (project author ran the verification cell, found this path, updated the copy cell to use it, and the subsequent dataloader sanity check succeeded: `Batch shape: torch.Size([16, 3, 256, 256])`, `Train patients: 151`, `Val patients: 33`).

**Rule going forward: never hardcode an assumed `/kaggle/input/...` path.** Always run a listing cell first (`os.listdir("/kaggle/input")`, recursing into whatever it finds) and read the actual output before writing the path into a copy/load step — this is exactly the failure mode that happened here: a plausible-looking guess based on "how Kaggle usually works" was wrong for this specific dataset attachment, and would have failed silently deep inside `dataset.py` (`FileNotFoundError` on `splits.csv`) rather than obviously, if not caught by an explicit verification step first.

## Working Kaggle cell recipe (verified, 2026-07-28)

This is the full sequence that got a real sanity check passing. Reusable for any future v2 (or later) Kaggle session against this same attached dataset.

1. **GPU check** — `torch.cuda.is_available()` / `torch.cuda.get_device_name(0)`, before anything else. Cheap, catches "forgot to enable GPU accelerator" immediately.
2. **Clone + `%cd`** — `!git clone https://github.com/manivafapour/eyes-defy-anemia.git` then `%cd eyes-defy-anemia`. Must be the magic `%cd`, not `!cd` — `!cd` only affects that one subprocess and doesn't persist to later cells; `%cd` does.
3. **List `/kaggle/input`** — see the finding above. Always run this before hardcoding a path.
4. **Copy (not symlink) the dataset into the repo** at `classification/data/processed/`, since `/kaggle/input/` is read-only and `classification/scripts/dataset.py`'s `PROCESSED_DIR` is computed relative to its own `__file__` location inside the cloned repo, not from any Kaggle-specific path. Confirmed working source path: `/kaggle/input/datasets/manivafapour33/processed-dataset`.
5. **`pip install -q optuna albumentations`** — the only two packages this pipeline needs that aren't already in Kaggle's base image. **Never** `pip install -r requirements.txt` from this repo on Kaggle — that file is pinned to the local Windows/CUDA 13.0 build (`torch==2.13.0+cu130`) and would try to reinstall Kaggle's own correctly-configured GPU PyTorch with an incompatible build.
6. **Dataloader sanity check** — import `get_dataloaders` from `classification/scripts/dataset.py` directly and pull one real batch, before running any real (slow) training script. Catches a data-path problem in seconds instead of after minutes of wasted GPU time on trial 0 of a real run. Verified working: `Batch shape: torch.Size([16, 3, 256, 256])`, `Train patients: 151`, `Val patients: 33`.
7. **Run the actual `classification/v2_scripts/train_*_v2.py` script(s).**

## "Save Version -> Save & Run All" (background/headless execution)

This is the project author's chosen workflow for actually running training: queue multiple `!python classification/v2_scripts/train_*_v2.py` cells, then use Kaggle's "Save & Run All" to execute the whole notebook unattended in a fresh kernel, so the tab can be closed. Behavioral notes that matter for this mode specifically:

- **Every cell must be fully non-interactive and correct on the first pass** — there's no human in the loop to read a diagnostic cell's output and manually adjust a path partway through, unlike the earlier interactive debugging session that found the path issue above. Once a path/config is confirmed, it should be hardcoded directly into the cell, not left as "check this and adjust."
- **A failed `!python ...` subprocess (non-zero exit code) does NOT halt the rest of a Jupyter "Run All" by default** — IPython's `!` shell escape doesn't raise on a non-zero exit unless something explicitly checks `_exit_code`. This means if one queued script crashes (e.g. an OOM on a heavier architecture), the notebook keeps going to the next queued script rather than stopping the whole chain — good for resilience across a long queue, but it also means "the notebook finished running" is not proof every script in it succeeded. **Always check each script's own output/log for errors after a Save & Run All completes, not just whether the notebook run itself completed.**
- **Per-script output is only fully written at the END of that script's own `run_study()` call** (`{model_name}_trials.csv`, `{model_name}_study_summary.json`, and all plots are written once, after all `n_trials` for that combo complete). The checkpoint (`.pth`) can save earlier/incrementally (whenever a trial finds a new best within its own loop), but the log/plot artifacts cannot. Practical implication: if a session gets killed or hits a limit partway through script N in a queued chain, scripts `1..N-1` have their full outputs safely on disk already; script `N` itself will have at most a partial checkpoint and no logs/plots at all. **This is why queue order should go cheapest/most-certain-to-finish first** — a truncation then loses only the last (most expensive, least-likely-attempted) item, not the whole batch.

## Batching decision: light/medium architectures vs. heavy transformers (2026-07-28)

Kaggle GPU session/quota limits are real but not something this session has precise, verified numbers for (the pilot test that would have measured this was explicitly skipped, per the project author's decision recorded in `02_current_status.md`). Rather than guess a total time budget for chaining all 18 combos into one session, split by weight class instead, consistent with the roadmap's existing "schedule by weight class" plan:

- **Batch 1 (this session):** all 7 light/medium architectures × 2 tissue types = **14 combos** — `regnet_y_400mf`, `mobilenet_v3_small`, `efficientnet_b0`, `resnet18`, `densenet121`, `convnext_tiny`, `swin_t` (all comfortably under 30M params). Queued cheapest-first (`regnet_y_400mf` first) so a truncation, if it happens, loses the least.
- **Batch 2 (separate session, not yet run):** `vit_b_16` and `vit_l_16` × 2 tissue types = **4 combos**. These are the two genuinely heavy architectures (86.6M and 304.3M params respectively) with meaningfully higher per-forward-pass cost than everything else in the roster — deliberately kept out of batch 1 to avoid a long chain's most expensive items being the ones most likely to get cut off by a session limit.

Status as of this entry: batch 1's 14 combos have been queued into the Kaggle notebook (exact scripts recorded in `01_roadmap.md`/`02_current_status.md`) but **not yet confirmed complete** — this file will be updated once real results are pulled back.

## Notebook fix: explicit, robust output persistence (2026-07-28)

Project author reported the notebook "does not save output artifacts properly." Diagnosis: it isn't strictly true that nothing gets saved — `trainer_engine.py`'s `CHECKPOINTS_DIR`/`LOGS_DIR`/`PLOTS_DIR` already resolve to `classification/outputs/{checkpoints,logs,plots}/` relative to wherever the repo is cloned, i.e. `/kaggle/working/eyes-defy-anemia/classification/outputs/` on Kaggle — which *is* under `/kaggle/working/` and *should* get captured by Kaggle's own "Save Version" snapshot mechanism. But two real, concrete problems existed even so:

1. **No single, obvious, top-level download location** — results were nested 3 directories deep (`eyes-defy-anemia/classification/outputs/...`), not at `/kaggle/working/` itself, making them easy to miss or tedious to collect by hand across 14 combos.
2. **No robustness to a mid-run interruption.** If the run got cut short partway through the 14 queued combos (a real possibility already flagged in this file, given no pilot test was run), whatever completed so far would still exist on disk but with no single consolidated, downloadable snapshot reflecting exactly what had finished.

**Fix applied to `classification/Kaggle-Notebook/classification-final-fixed.ipynb`:** added a `sync_outputs()` helper (defined once, right after the "## Training" header) that copies `classification/outputs/{checkpoints,logs,plots}/` into a clean top-level `/kaggle/working/outputs/` and zips it to `/kaggle/working/batch1_results.zip`. Critically, **`sync_outputs()` is called after every one of the 14 training cells, not just once at the end** — so a mid-run interruption still leaves a complete, correctly up-to-date consolidated snapshot of whatever finished, not nothing. A final summary cell prints the consolidated directory's contents and the zip's size once the notebook completes.

**Verified before considering this done, not just written:**
- `nbformat.validate()` passed (strict schema check) after patching, same standard as the original notebook generation.
- Confirmed `shutil.make_archive` doesn't crash on an empty source directory (the very first no-op `sync_outputs()` call, right after its own definition, runs before any training has produced anything) — tested directly, produces a valid empty zip rather than raising.
- Full end-to-end simulation with real files (fake checkpoint + fake log JSON) confirmed the copy+zip logic produces a correct archive with the expected internal paths (`checkpoints/...`, `logs/...`), not just checked in the abstract.

Total cell count: 24 -> 27 (1 new helper-definition cell, 14 existing training cells each gained one appended `sync_outputs()` line, 2 new cells at the end).

## Notebook review (`classificatio-final.ipynb`, before batch 1 was launched, 2026-07-28)

Reviewed the actual saved Kaggle notebook (cells 0-7) against the specified recipe before the project author hit Save & Run All. Two real findings:

**Blocking: the `pip install -q optuna albumentations` cell never made it into the notebook.** Cells 0-6 (GPU check, clone, input-path discovery, dataset copy, dataloader sanity check) are all present and their real output confirms they work — but there is no cell anywhere that installs `optuna`. `trainer_engine.py` does `import optuna` at module level, so the very first `!python classification/v2_scripts/train_*_v2.py` cell would fail immediately with `ModuleNotFoundError: No module named 'optuna'`, and since a failed `!python` cell doesn't halt "Run All" (see above), all 14 queued scripts would likely fail the same way in sequence rather than just the first one. **Lesson: verify the pip-install cell landed in the actual saved notebook, don't assume a "here are the cells" spec was followed exactly** — it's an easy step to drop when manually assembling cells from a chat into a real notebook, and nothing before the first training script would have surfaced its absence (the dataloader sanity check only exercises `dataset.py`'s imports — `albumentations`/`pandas`/`PIL`/`torch` — never `trainer_engine.py`'s, so it passing is not proof optuna is installed).

**Cleanup, not blocking:** the notebook's cell 4 used `SRC_DIR = Path("/kaggle/input/datasets/manivafapour33")` — one path segment too shallow (missing `/processed-dataset`) — so it copied the entire `processed-dataset` folder as a single unit into `classification/data/processed/processed-dataset/` (431 files, wrongly nested). Cell 5 immediately after has the correct full path and populated the real `classification/data/processed/images/` correctly (428 files = 217 palpebral + 211 forniceal_palpebral, matching the known dataset exactly) — so the end state is functionally correct (confirmed by cell 6's passing sanity check), but cell 4's mistaken output is dead clutter left sitting on disk. Recommended: delete cell 4 entirely, keep only cell 5.

**Minor, cosmetic:** cell 1's printed cwd after `%cd eyes-defy-anemia` was `/kaggle/working/eyes-defy-anemia/eyes-defy-anemia` (doubly nested) — evidence this cell was re-run at least once within the same kernel session (consistent with the cell-4-mistake-then-cell-5-fix pattern, i.e. this was an interactive debugging pass, not a single clean top-to-bottom run). Should not recur on a genuine fresh "Save & Run All" (starts from a clean `/kaggle/working/`), but added robustness recommendation: prefix the clone with `!rm -rf eyes-defy-anemia` so a re-run within the same kernel stays idempotent instead of nesting.

## Corrected notebook generated (2026-07-28)

All three findings above (missing `optuna` install, the shallow-path/duplicate data-copy cells, plus the nested-clone robustness fix) were rolled into a single regenerated notebook: **`classification/Kaggle-Notebook/classification-final-fixed.ipynb`** (local only, not committed to git — generated on request, not pushed). 24 cells total (4 markdown section headers + 20 code), in order: GPU check → `rm -rf` + clone + `%cd` → `/kaggle/input` listing (diagnostic) → `pip install -q optuna albumentations` → single clean data-copy cell (`shutil.rmtree(DST_DIR, ignore_errors=True)` before copying, replacing the old two-cell shallow-path-then-fix pattern — fully idempotent, no more stray `processed-dataset/` leftover) → dataloader sanity check → 14 batch-1 training cells (`regnet_y_400mf` → `mobilenet_v3_small` → `efficientnet_b0` → `resnet18` → `densenet121` → `convnext_tiny` → `swin_t`, each × palpebral then forniceal_palpebral, cheapest first). Validated two ways before confirming: parsed as JSON, and passed `nbformat.validate()` (strict schema check, not just "is it valid JSON") — no cells have been executed (this file was generated, not run; `execution_count: null` and empty `outputs` on every code cell, honestly reflecting that).
