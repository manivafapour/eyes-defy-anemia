# Current Status — MLOps / Deployment (`app/`)

Last updated: 2026-08-05

## Where things stand
Full stack built and **live-verified end-to-end (2026-08-05)**. Real ConvNeXt-Tiny
classifier + mock segmenter behind FastAPI; React+Vite SPA served by FastAPI at `/`.
Confirmed over HTTP: `/api/health` ok; `/api/version` (classifier `is_mock=false`,
segmenter `is_mock=true`, cpu); `/api/predict` runs the real model on an uploaded image
(returned label=anemic, p=0.626 on a mock-crop test, + base64 crop_preview + mock warning);
the SPA renders (Demo-mode banner + upload UI). Serving deps (fastapi, uvicorn,
pydantic-settings, python-multipart) were pip-installed into the training `venv/` —
torch 2.13.0+cu130 confirmed untouched (pydantic v2 already present via albumentations 2.0.8).

## What exists on disk (verified 2026-08-05)
- **Stage 2 model READY:** `classification/v2_clean_scripts/outputs/convnext_tiny_palpebral_v2_clean/best_convnext_tiny_palpebral_v2_clean.pth`
- **Stage 1 model MISSING:** only crop-based seg checkpoints (`best_{unet,attention_unet,resunet}.pth`); **zero aligned checkpoints**. The crop-based U-Net does NOT generalize to raw photos → unusable for production. Aligned segmenter must be trained (`Segmentation/` infra ready, never run).

## Active backends
- **Segmenter:** `MockSegmenter` — fixed lower-eyelid ellipse (~8% coverage), ignores image content. Default `mock`; real `AlignedUNetSegmenter` pending training.
- **Classifier:** `ConvNeXtTinyClassifier` — **REAL, wired + parity-verified 2026-08-05.** Default backend is now `convnext_tiny`; `MockClassifier` still available via `EYESDEFY_CLASSIFIER_BACKEND=mock`. Weights copied to `app/weights/` (gitignored, 107 MB).

## Stage 2 parity: PROVEN (2026-08-05)
`app/tests/test_convnext_parity.py` rebuilds `ConvNeXtTinyClassifier` and runs it over the
palpebral val split; it reproduces the committed `study_summary.json` metrics **exactly** —
overall CM `[[17,2],[0,14]]`, F1 0.9333, India `[[4,0],[0,10]]`, Italy `[[13,2],[0,4]]`.
Architecture reconstructed from `trainer_engine.build_convnext_tiny` (frozen ConvNeXt-Tiny +
`Sequential(Dropout(0.2), Linear(768,1))` head, `weights=None` + strict state_dict load);
eval transform mirrors `dataset.get_eval_transforms` (albumentations `Resize(256)`→`Normalize`
→`ToTensorV2`, pinned `albumentations==2.0.8`); threshold `> 0.5`. No train/serve skew in Stage 2.

## Remaining risk: crop-synthesis parity at the seg→cls seam
Stage-2 *classifier* parity is now proven (above). The still-open risk is the *crop
synthesis*: `app/core/preprocessing.py::synthesize_crop` builds the classifier's input from
the raw photo + Stage-1 mask, which is a different distribution from the human-made source
crops the classifier trained on. This can only be tested once the real segmenter exists
(the mock returns a fixed ellipse). Golden-test synth crops vs. the 217 archive source crops
then. The white-background bug recurred twice from duplicated preprocessing; serving is the
3rd risk site — hence one source of truth in `preprocessing.py`.

## Immediate next steps
1. ~~Wire real ConvNeXt-Tiny + parity test~~ — **DONE 2026-08-05** (see Stage 2 parity above).
2. ~~React + Vite frontend~~ — **DONE 2026-08-05** (built + live-verified end-to-end).
3. **Deploy — pivoted 2026-08-05:** HF Spaces dropped (Docker SDK now paid) → target is **Render.com / Railway** free Docker tier. `Dockerfile` reusable as-is (`${PORT:-7860}`). Current focus is **local testing**. Frontend converted to **Tailwind CSS v4** (verified: builds, CSS 16.65 kB, served DOM renders all components incl. the mock banner).
4. (Parallel ML track) train + verify the aligned segmenter, then swap `MockSegmenter` → `AlignedUNetSegmenter`; then add the crop-synthesis parity test.
