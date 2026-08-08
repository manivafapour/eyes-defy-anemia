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

## Still open

- Whether the training scripts themselves run cleanly end-to-end on Kaggle's T4×2 (data staging is now confirmed working; no training has been confirmed successful yet as of this entry).
- Whether the Strong-tier combos (Swin-Large at 512×512, batch_size=16) fit in the T4's 16GB — flagged as a watch-item in the notebook itself (`04_pretrained_architecture_sweep.md`'s hardware-constraint note), not yet confirmed either way on real Kaggle hardware.
