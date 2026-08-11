"""FastAPI dependency providers."""
from __future__ import annotations

from fastapi import Request

from app.core.pipeline import InferencePipeline


def get_pipeline(request: Request) -> InferencePipeline:
    """Return the process-wide pipeline loaded once at startup (see main.lifespan)."""
    return request.app.state.pipeline
