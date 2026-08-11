# Eyes-Defy-Anemia — Serving App (`app/`)

Two-stage anemia screening API: **raw eye photo → segment (Stage 1) → classify (Stage 2) → Anemic / Non-Anemic**.

Currently runs on **mock backends** (Option C): the platform + inference pipeline are
complete; the real ConvNeXt-Tiny (Stage 2) and aligned U-Net (Stage 1) drop in behind
the existing interfaces without changing pipeline or API code.

## Quickstart

```bash
pip install -r app/requirements.txt
uvicorn app.main:app --reload      # run from the repo root
```

- Interactive docs: http://localhost:8000/docs
- Health: `GET /api/health` · Version/backends: `GET /api/version`
- Predict: `POST /api/predict` with `multipart/form-data` field `file` (an eye photo)

## Layout

| Path | Role |
|---|---|
| `core/pipeline.py` | Framework-agnostic two-stage `InferencePipeline` (the heart) |
| `core/preprocessing.py` | **Parity-critical** crop synthesis — single source of truth |
| `core/exceptions.py` | Domain errors (mapped to HTTP by the API layer) |
| `models/base.py` | `BaseSegmenter` / `BaseClassifier` interfaces + result types |
| `models/segmentation.py` | `MockSegmenter` (+ `AlignedUNetSegmenter` swap-in point) |
| `models/classification.py` | `MockClassifier` + `ConvNeXtTinyClassifier` (Phase-1 stub) |
| `models/factory.py` | Settings → concrete backends |
| `api/routes.py` | `/health`, `/version`, `/predict` |
| `main.py` | FastAPI app, lifespan model loading, CORS |
| `config.py` | `pydantic-settings` (env prefix `EYESDEFY_`) |

## Switching backends

```bash
# once ConvNeXt-Tiny is wired + parity-tested:
EYESDEFY_CLASSIFIER_BACKEND=convnext_tiny uvicorn app.main:app
```

Weights go in `app/weights/` (gitignored). Nothing here is a medical device — every
response carries a screening disclaimer.
