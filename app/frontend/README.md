# Frontend — React + Vite SPA

Upload an eye photo → `POST /api/predict` → result card (label, P(anemic) with threshold
marker, the crop the model saw, warnings, disclaimer).

## Dev

```bash
npm install          # from app/frontend/
npm run dev          # http://localhost:5173  (proxies /api -> :8000)
```

Run the API in another terminal (`uvicorn app.main:app --reload` from the repo root).
API calls use relative `/api` paths, so the same code works in prod without changes.

## Build (for single-container deploy)

```bash
npm run build        # emits app/frontend/dist/
```

When `app/frontend/dist/` exists, `app/main.py` serves it at `/` automatically, so the
whole app (SPA + API) is one FastAPI process on one origin — ready for Docker / HF Spaces.
