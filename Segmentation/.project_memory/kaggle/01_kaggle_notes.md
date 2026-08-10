# Kaggle Execution Notes — Segmentation (9-architecture pretrained sweep)

Dedicated log for everything specific to running `segmentation-pretrained-sweep.ipynb` on Kaggle (paths, environment quirks, workflow patterns) — kept separate from `04_pretrained_architecture_sweep.md`, which covers the pipeline/architecture/results side, not the execution environment. Created 2026-08-08, mirroring `classification/.project_memory/kaggle/01_kaggle_notes.md`'s own split (per this project's rule 11 in `03_tech_stack_and_rules.md`: give a distinct, growing topic its own file rather than letting one file absorb everything).

## Confirmed: real dataset mount path, two levels deeper than the naive guess

**What was assumed (wrong, first attempt):** the notebook originally had a single `KAGGLE_DATASET_DIR` placeholder, implicitly assuming one combined dataset mounting at something like `/kaggle/input/<slug>/`.

**What actually happened:** the project author uploaded the two zips as **two separate Kaggle datasets** (`aligned_raw` and `aligned_raw_forniceal`), and the real mount structure — confirmed via a recursive `/kaggle/input` listing, not assumed — turned out to be doubly-nested:

```
/kaggle/input/
  datasets/
    manivafapour21/                    <- Kaggle username
      aligned-raw/                     <- Kaggle's slug for the "aligned_raw" dataset (hyphenated)
        aligned_raw/                   <- the ZIP's OWN internal top-level folder, preserved as-is
          alignment_log.csv
          images/
          masks/
      aligned-raw-forniceal/           <- Kaggle's slug for the "aligned_raw_forniceal" dataset
        aligned_raw_forniceal/         <- same story, zip's own internal folder preserved
          alignment_log.csv
          images/
          masks/
```

This is the **same class of gotcha** classification's own Kaggle notes already documented (`/kaggle/input/datasets/<username>/<slug>/` instead of `/kaggle/input/<slug>/` directly) — confirms that pattern generalizes across this project's Kaggle datasets — **plus one extra level of nesting specific to segmentation's zips**: because `aligned_raw.zip`/`aligned_raw_forniceal.zip` were built with their own name as an internal top-level folder (`aligned_raw/images/...`, not `images/...` at the zip root — see `Segmentation/scripts/build_aligned_dataset{,_forniceal}.py`), and each was uploaded as its own single-zip dataset, Kaggle preserved that internal folder intact rather than flattening it. Classification's own dataset didn't hit this second level because its zip's contents were flat at the zip root to begin with.

**Confirmed working values** (real project author screenshot, 2026-08-08):
```python
ALIGNED_RAW_DATASET_DIR = "/kaggle/input/datasets/manivafapour21/aligned-raw"
ALIGNED_RAW_FORNICEAL_DATASET_DIR = "/kaggle/input/datasets/manivafapour21/aligned-raw-forniceal"
```
Pass the dataset's own mount root (one level *above* the doubly-nested `aligned_raw/`/`aligned_raw_forniceal/` folder) — `stage_tissue_data()` in the notebook already looks for `dataset_dir / name` itself, so it finds the inner folder automatically. This is exactly the "Case 1: nested" branch of that function's 3-way fallback, verified locally against the real zip contents before this was ever run for real on Kaggle.

**Rule going forward, restated from classification's own notes but now doubly proven:** never hardcode or guess a `/kaggle/input/...` path for this project, even from a plausible-looking single-level listing. The original listing cell in this notebook only went 2 levels deep and missed this — see the fix below.

## Notebook fixes applied in response (2026-08-08)

1. **Single `KAGGLE_DATASET_DIR` → two independent variables** (`ALIGNED_RAW_DATASET_DIR`, `ALIGNED_RAW_FORNICEAL_DATASET_DIR`), since the two zips ended up as two separate Kaggle datasets with two different mount paths, not one combined dataset as originally instructed/assumed.
2. **`stage_tissue_data()` now tries 3 layouts, not 2:** (a) `dataset_dir/{name}/images,masks/` — the actual case here, zip's own folder preserved; (b) `dataset_dir/images,masks/` directly — Kaggle flattened it; (c) `dataset_dir/{name}.zip` — never auto-extracted. All three verified locally against the real zip files (three fake directory trees built to match each case) before trusting it, not just reasoned through.
3. **The `/kaggle/input` listing cell itself was too shallow** — only recursed 2 levels, which is exactly why the first real run's screenshot only showed `datasets -> ['manivafapour21']` and not the actual dataset slugs underneath. Replaced with a recursive `print_tree()` (depth 4) so the full structure is caught in one pass. Verified against a fake directory tree matching the real observed structure before shipping.

All three fixes committed and pushed (commits `de8e3b2`, `c03397e`) before the project author's second real attempt, which is what produced the confirmed working paths recorded above.

## Second, distinct placeholder bug: confirmed paths were documented but never applied to the notebook itself (2026-08-08)

Despite the real mount paths being confirmed and recorded above, the project author's actual first "Run All" attempt on Kaggle still hit the exact same class of error:

```
FileNotFoundError: Could not find aligned_raw/, images+masks/, or aligned_raw.zip under
/kaggle/input/REPLACE_WITH_ACTUAL_PATH_FROM_LISTING_ABOVE -- run the /kaggle/input listing
cell above and check what's actually there.
```

**Root cause, distinct from the first bug above:** the committed notebook's dataset-path cell had never actually been edited to use the confirmed values — they existed only as prose in this memory file, not in the executable cell itself, which still shipped with the literal `REPLACE_WITH_ACTUAL_PATH_FROM_LISTING_ABOVE` placeholder. Not a new path-discovery problem; a gap between "we recorded the answer" and "the artifact that runs was updated with it."

**Fix:** the cell now hardcodes the confirmed values directly as the default, for this project author's Kaggle account, rather than leaving a placeholder that must be manually edited every time the notebook is opened:
```python
ALIGNED_RAW_DATASET_DIR = "/kaggle/input/datasets/manivafapour21/aligned-raw"
ALIGNED_RAW_FORNICEAL_DATASET_DIR = "/kaggle/input/datasets/manivafapour21/aligned-raw-forniceal"
```
`Segmentation/Kaggle-Notebook/segmentation-pretrained-sweep.ipynb`, committed `691cb7a`, pushed. A comment in the cell notes these are confirmed-for-this-account values and to re-run the `print_tree()` listing cell and update them if the datasets are ever re-attached under a different username/slug.

Note this fix updates the **repo copy** of the notebook only — a live Kaggle notebook is a separate, independently-edited copy that does not pick up a GitHub change automatically. The project author applied the same two lines directly in their live Kaggle session to unblock the run in progress.

## Confirmed via real Kaggle execution, first time this notebook has actually run (2026-08-08)

After the fix above, the project author's data-staging cell succeeded end-to-end on real Kaggle infrastructure:

```
aligned_raw: 201 images, 201 masks staged at Segmentation/data/processed/aligned_raw
aligned_raw_forniceal: 211 images, 211 masks staged at Segmentation/data/processed/aligned_raw_forniceal
```

These counts match exactly the locally-verified alignment totals (201/217 palpebral successes, 211/217 forniceal_palpebral successes — `CLAUDE.md` §1.4.2/§1.4.4 and `04_pretrained_architecture_sweep.md`'s forniceal section) — real confirmation the correct files reached Kaggle, not just a non-error. This is the first time any part of this notebook has been confirmed running on actual Kaggle hardware, as opposed to structural/local-only verification.

## Third bug: all 18 combos failed identically with `ModuleNotFoundError: No module named 'models'` (real Kaggle "Version 3" run, confirmed via baked-in notebook output)

The Kaggle-auto-pushed "Version 3" of the notebook (commit `3819cbe`, merged into this repo at `f40f503`) preserved its own real execution output, and it shows **every one of the 18 training cells failing identically**:
```
Traceback (most recent call last):
  File ".../Segmentation/scripts/train_pretrained/train_....py", line 16, in <module>
    from models.segmentation.pretrained_registry import ARCHITECTURE_REGISTRY  # noqa: E402
ModuleNotFoundError: No module named 'models'
[sync_outputs] 0 files consolidated under /kaggle/working/outputs, zipped to /kaggle/working/segmentation_sweep_results.zip
```
This is the `sys.path` bug already diagnosed and fixed in commit `4816cee` (`04_pretrained_architecture_sweep.md` has the root-cause writeup: the entry-point script template only added `Segmentation/scripts/` to `sys.path`, not `Segmentation/` itself, so `dataset`/`trainer_engine` resolved but `models.segmentation.pretrained_registry` never did). This Version 3 run is the direct evidence that the bug was real and total — 0/18 combos produced any output before the fix landed, confirming the project author's observation ("None of the models go through training") exactly.

## Fourth bug: disk quota exceeded mid-sweep, after the sys.path fix (real Kaggle "Version 4" run)

After the sys.path fix (`4816cee`/`f40f503`), the project author's next "Save & Run All" (Version 4) actually started training for real — it ran **5h40m (20422s)** before Kaggle killed it with **"Your notebook tried to use more disk space than is available"** (Output size at failure: 20.93 GB). Confirms the fix worked (training real progress, not an import crash) but surfaced a second, independent problem.

**Root cause:** each of the 18 training scripts (`trainer_engine.py`'s `_save_outputs()`) can persist up to 3 full fp32 checkpoint files per combo — `best_{model_name}.pth` plus one `best_{model_name}_{loss_fn}.pth` per loss function (`bce_dice`, `focal_tversky`) — and none of these are ever deleted between combos; they accumulate in `Segmentation/outputs/checkpoints/`. For the larger architectures this is substantial: ConvNeXt-Large (~203M params) ≈ 775MB/file, Swin-Large (~234M) ≈ 890MB/file, Swin-Base and TransUNet (~121M each) ≈ 460MB/file. On top of that, the notebook's `sync_outputs()` (called after every combo, by design, for crash-resilience) **copied** that entire accumulating folder into `/kaggle/working/outputs/` and left the source in place, then zipped the copy into a third representation — so at any point roughly 3 copies of everything produced so far existed on disk simultaneously. Summing all 18 combos' worst-case checkpoint sizes with this 3x multiplier comfortably exceeds a ~20GB quota partway through the Mid/Strong tiers, matching a mid-run (not immediate) failure.

**Fix (project author chose "stop triplicating output" over dropping per-loss checkpoints or switching to fp16):** `sync_outputs()` now deletes `Segmentation/outputs/{checkpoints,logs,plots}/` immediately after copying it into `/kaggle/working/outputs/` (`shutil.rmtree(src)` right after `shutil.copytree(...)`). This is safe because every training script recreates those directories fresh via its own `mkdir(parents=True, exist_ok=True)` on its next run — nothing depends on the old source persisting. Cuts total footprint from ~3x to ~2x (the `/kaggle/working/outputs/` mirror + the zip made from it) with no loss of any file or feature. Not yet re-run on Kaggle to confirm this is sufficient headroom for all 18 combos to complete — if it still runs out partway through the Strong tier, the two declined options (drop per-loss checkpoint weights, or save in fp16) remain available.

## Full disk-usage fix: drop per-loss checkpoints + fp16 + split into 3 tier-specific notebooks (2026-08-09)

The single `sync_outputs()` fix above (source deleted after mirroring, ~3x down to ~2x) was judged insufficient on its own: total *unique* checkpoint data across all 18 combos, if every combo writes all 3 checkpoint variants (best-overall + one per loss function) at fp32, comes to **≈20.9GB by itself** (summed from the 9 architectures' measured param counts × 4 bytes × 3 variants × 2 tissue types) — suspiciously close to the exact 20.93GB crash figure, and even at the improved ~2x multiplier that's ~42GB at the very end of a full run, likely still over quota. Project author chose to implement all three previously-discussed mitigations together rather than pick one:

1. **Dropped per-loss-function checkpoint weight files entirely** (`trainer_engine.py`'s `make_objective`/`_save_outputs()`) — only `best_{model_name}.pth` (the overall best) is now saved; the `best_{model_name}_{loss_fn}.pth` files and the `summary["per_loss_fn_checkpoints"]` JSON key are gone. The per-loss-function *comparison* (trial count, mean/max Dice per loss) is unaffected — it's computed from `trials_df.groupby("params_loss_fn")`, not from the checkpoint files, so the bce_dice-vs-focal_tversky analysis this was originally built for (§3.2b) is fully intact.
2. **Checkpoints now saved in fp16, not fp32** — new `_half_state_dict()` helper in `trainer_engine.py` casts only floating-point tensors (leaves integer buffers like BatchNorm's `num_batches_tracked` untouched) before `torch.save`. Halves every checkpoint file's size. Verified locally with a real save/load round-trip test (tiny Conv+BatchNorm model): dtypes correctly halved on save, `load_state_dict` into a fresh fp32 model works without any special-casing (`Tensor.copy_` casts across dtypes automatically), values preserved within fp16 precision, integer buffer preserved exactly.
3. **Split the single 18-combo notebook into 3 independent tier-specific notebooks** — `segmentation-pretrained-sweep-{base,mid,strong}.ipynb` (`Segmentation/Kaggle-Notebook/`), each a full standalone notebook (Setup/Data/sanity-check cells duplicated, then only that tier's 6 training cells). Running each tier as its own Kaggle session means peak disk usage at any point is bounded by ~6 combos' worth of data, not ~18. The original combined notebook is kept (not deleted) as the historical record of the real crash, with a superseded-notice markdown cell added at the top pointing to the 3 new ones.

**Both trainer_engine.py changes verified via a real 2-trial/1-epoch dry run** (not just read/reasoned about): exactly one checkpoint file produced (17.861MB for EfficientNet-B1 — matches the expected fp16 size, 8,757,105 params × 2 bytes ≈ 17.5MB, almost exactly), no per-loss checkpoint files, final test-set evaluation correctly loaded the fp16 checkpoint back into a fresh fp32 model and computed all metrics normally, summary JSON confirmed missing `per_loss_fn_checkpoints` but still containing a correct `per_loss_fn_comparison` table, all 6 plots still generated. Dry-run artifacts deleted afterward per convention.

**All 3 new notebooks validated** via `nbformat.validate()` (22-23 cells each) and a check that every `!python ...`-referenced script path actually exists on disk — all 18 combos accounted for exactly once across the 3 files (Base: combos 1-6, Mid: 7-12, Strong: 13-18), none duplicated or missing.

**Revised disk math with all 3 fixes combined** (1 fp16 checkpoint per combo, ~2x sync_outputs multiplier, split into thirds): Base tier ≈0.5GB, Mid tier ≈1.9GB, Strong tier ≈4.4GB peak — all comfortably under a ~20GB quota with wide margin, versus the original design's ~42-63GB worst case for the full 18-combo run. This has not yet been confirmed on a real Kaggle run.

## Still open

- Whether the 3-notebook split + fp16 + single-checkpoint fixes actually get every tier through to completion on real Kaggle hardware — not yet run.
- Whether the training scripts themselves produce sensible results end-to-end on Kaggle's T4×2 (no combo has yet completed a full 12-trial study to confirm real training quality, independent of the infrastructure bugs above).
- Whether the Strong-tier combos (Swin-Large at 512×512, batch_size=16) fit in the T4's 16GB **GPU** memory (separate from the disk-quota issue above) — flagged as a watch-item in the notebook itself (`04_pretrained_architecture_sweep.md`'s hardware-constraint note), not yet confirmed either way on real Kaggle hardware.
