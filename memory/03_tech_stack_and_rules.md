# Tech Stack & Rules — MLOps / Deployment (`app/`)

## Locked decisions (user-confirmed 2026-08-05)
- **Backend:** FastAPI (async, Pydantic, auto-OpenAPI). Not Django.
- **Frontend:** React + Vite SPA, **Tailwind CSS v4** (`@tailwindcss/vite` plugin; converted from plain CSS 2026-08-05).
- **Serving:** CPU inference (models are small; no GPU). Both backends load once via
  `lifespan` into `app.state` — never per request (the anti-bottleneck decision).
- **Deploy target:** **Render.com / Railway** (free Docker tiers). **HF Spaces dropped 2026-08-05**
  — its Docker SDK is now a paid feature. Single container: FastAPI serves the built React static
  bundle **and** `/api`; binds `${PORT:-7860}` (Render/Railway inject `$PORT`), so the existing
  `Dockerfile` is reusable as-is. Local dev is the current focus.
- **Serving classifier champion:** ConvNeXt-Tiny / palpebral (28.6M params, F1 0.9333,
  AUC 0.940, 256px input) — best accuracy *and* deployment-friendly size.
- **Near-term goal:** thesis demo ASAP. Defer CI/CD, monitoring, scaling, ONNX.

## Architecture rules
1. **Core is framework-agnostic.** `app/core` + `app/models` import NO FastAPI. The API
   is a thin wrapper, so the pipeline is testable/reusable from CLI, notebook, batch.
2. **Backends behind ABCs** (`BaseSegmenter` / `BaseClassifier`). Swap mock→real via
   `EYESDEFY_*_BACKEND` env — zero pipeline/API changes.
3. **ONE preprocessing source of truth** (`app/core/preprocessing.py`). Parity-critical;
   golden-test it (Phase 0). Never duplicate crop logic (it caused the same bug twice).
4. **Config via `app/config.py`** (`pydantic-settings`, prefix `EYESDEFY_`). Nothing hardcoded.
5. **Never commit weights** (`app/.gitignore`: `weights/*.pth`). Bake into the image at deploy.
6. **Output is a SCREENING result, not a diagnosis** — disclaimer in every response.
7. This phase is isolated like `Segmentation/` and `classification/`: `app/` only *imports*
   from `classification/` (the model builder) — it does not edit training code.

## Run
```
pip install -r app/requirements.txt
uvicorn app.main:app --reload      # from repo root
# docs: http://localhost:8000/docs
```
