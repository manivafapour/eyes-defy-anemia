# Roadmap — MLOps / Deployment Phase (`app/`)

Serve the two-stage seg→cls pipeline as a web app. `[x]` = done & verified.
Scoped for a **thesis demo ASAP** (Option C: build the platform now, mock Stage 1,
swap the real aligned segmenter in later). Started 2026-08-05.

## Phase 0 — Model readiness & contract
- [ ] Train + verify the aligned raw-photo segmenter (`Segmentation/`, pending Kaggle run) — the real Stage-1 blocker, **not** an app task
- [ ] Freeze champions: ConvNeXt-Tiny/palpebral (Stage 2, on disk) + winning aligned U-Net (Stage 1, pending)
- [ ] Golden parity test: synthesized crop vs. the 217 archive source crops
- [ ] `model_bundle.json` manifest (ids, weight hashes, thresholds, input sizes)

## Phase 1 — Inference core (`app/core`, `app/models`)
- [x] Framework-agnostic `InferencePipeline` + `Base{Segmenter,Classifier}` interfaces + mock backends (scaffolded 2026-08-05)
- [x] Shared parity preprocessing (`synthesize_crop`)
- [x] FastAPI app runnable end-to-end on mocks
- [x] Wire real ConvNeXt-Tiny (`ConvNeXtTinyClassifier`) — reproduces committed val metrics exactly (2026-08-05)
- [x] Golden parity test (`app/tests/test_convnext_parity.py`) — reproduces `study_summary` val CM/F1 bit-for-bit
- [ ] More unit tests (pipeline quality gate, preprocessing edge cases)

## Phase 2 — API (`app/api`, `app/main.py`)
- [x] Lifespan model loading; `/api/health`, `/api/version`, `/api/predict`; CORS; mask quality gate; screening disclaimer (scaffolded 2026-08-05)
- [ ] Request/prediction logging (for drift/confound monitoring)

## Phase 3 — Frontend (React + Vite)  — DONE (built + live-verified 2026-08-05)
- [x] React+Vite SPA in `app/frontend/` — upload panel + result card (P(anemic) bar with threshold marker, crop-used preview, warnings, disclaimer)
- [x] Builds to `dist/` (verified); `app/main.py` serves it at `/` (single origin, no CORS in prod)
- [x] Full stack verified LIVE over HTTP: `/api/{health,version,predict}` + SPA render, real ConvNeXt on a mock crop

## Phase 4 — Containerization  — DONE (written + reviewed 2026-08-05; not build-verified, no Docker locally)
- [x] Multi-stage `Dockerfile` (node:20 build → python:3.11-slim runtime, CPU torch from wheel index, weights baked, non-root UID 1000, binds `${PORT:-7860}`)
- [x] `.dockerignore` (excludes Segmentation/classification/Source/venv + node_modules/dist)
- [x] Fixed: deps install as root BEFORE dropping to non-root user (non-root can't write system site-packages)

## Phase 5 — Deploy (Render.com / Railway free tier)  — HF Spaces dropped 2026-08-05 (Docker SDK now paid)
- [x] `Dockerfile` is Render/Railway-ready as-is (`${PORT:-7860}`). `deploy/README_hf.md`/`DEPLOY.md` kept as the historical HF attempt.
- [ ] Deploy to Render/Railway (Docker) when ready — connect the repo, it builds the Dockerfile. Weights: bake in or pull at build.
- Current focus: local dev/testing (uvicorn + `npm run dev`, or build + single uvicorn).

## Deferred (post-demo)
CI/CD (GitHub Actions), Prometheus/Sentry, separate inference microservice, ONNX Runtime.
