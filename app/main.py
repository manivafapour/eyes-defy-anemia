"""FastAPI application entry point.

Run from the repo root::

    uvicorn app.main:app --reload

Loads both model backends once at startup into ``app.state`` (no per-request
reloads -- the key to avoiding the memory/latency bottleneck when chaining two
models), wires the API router under ``/api``, and enables CORS for the React + Vite
dev server.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import get_settings
from app.core.pipeline import InferencePipeline
from app.models.factory import build_classifier, build_segmenter

FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("eyesdefy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline once; hold both models resident for the process lifetime."""
    settings = get_settings()
    logger.info(
        "Loading models: segmenter=%s classifier=%s (device=%s)",
        settings.segmenter_backend,
        settings.classifier_backend,
        settings.device,
    )
    segmenter = build_segmenter(settings)
    classifier = build_classifier(settings)
    app.state.pipeline = InferencePipeline(
        segmenter,
        classifier,
        min_coverage=settings.min_mask_coverage,
        max_coverage=settings.max_mask_coverage,
    )
    logger.info("Pipeline ready: %s -> %s", segmenter.name, classifier.name)
    if segmenter.is_mock or classifier.is_mock:
        logger.warning("Running with MOCK backend(s) -- results are NOT clinically meaningful.")

    yield

    app.state.pipeline = None
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        summary="Two-stage (segmentation -> classification) anemia screening from eye photos.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api")

    # Single-container mode: once the React app is built (app/frontend/dist),
    # FastAPI serves it at "/" on the same origin as the API. Until then (dev),
    # expose a small JSON index and let the Vite dev server handle the UI.
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    else:
        @app.get("/", tags=["meta"])
        def root() -> dict[str, str]:
            return {"service": settings.app_name, "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
