"""Pydantic schemas -- the versioned HTTP contract for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app_version: str


class ModelInfo(BaseModel):
    backend: str
    is_mock: bool


class VersionResponse(BaseModel):
    app_name: str
    app_version: str
    segmenter: ModelInfo
    classifier: ModelInfo
    classifier_threshold: float
    device: str


class PredictionResponse(BaseModel):
    label: str = Field(description="'anemic' or 'non-anemic'")
    probability: float = Field(ge=0.0, le=1.0, description="P(anemic)")
    threshold: float
    is_anemic: bool
    is_mock: bool = Field(description="True if any stage used a mock backend")
    warnings: list[str] = Field(default_factory=list)
    crop_preview: str | None = Field(
        default=None,
        description="data:image/png;base64 of the exact crop the classifier saw",
    )
    segmenter_backend: str
    classifier_backend: str
    disclaimer: str


class ErrorResponse(BaseModel):
    detail: str
