"""HTTP routes (thin wrappers over the inference core)."""
from __future__ import annotations

import base64
import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from app.api.dependencies import get_pipeline
from app.config import Settings, get_settings
from app.core.exceptions import InvalidImageError, SegmentationQualityError
from app.core.pipeline import InferencePipeline
from app.core.preprocessing import load_image
from app.schemas.prediction import (
    HealthResponse,
    ModelInfo,
    PredictionResponse,
    VersionResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(app_version=settings.app_version)


@router.get("/version", response_model=VersionResponse, tags=["meta"])
def version(
    settings: Settings = Depends(get_settings),
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> VersionResponse:
    return VersionResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        segmenter=ModelInfo(backend=pipeline.segmenter.name, is_mock=pipeline.segmenter.is_mock),
        classifier=ModelInfo(backend=pipeline.classifier.name, is_mock=pipeline.classifier.is_mock),
        classifier_threshold=settings.classifier_threshold,
        device=settings.device,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    responses={
        400: {"description": "Invalid / unreadable image"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Image is not a usable eye photo"},
    },
)
async def predict(
    file: UploadFile = File(..., description="Full, uncropped eye photo (JPEG/PNG/WebP)"),
    settings: Settings = Depends(get_settings),
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> PredictionResponse:
    # --- validate content type ---
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type {file.content_type!r}. Allowed: {settings.allowed_content_types}.",
        )

    # --- read + size-limit ---
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_bytes}-byte limit.",
        )

    # --- decode ---
    try:
        image = load_image(data)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # --- run the (sync, CPU-bound) pipeline off the event loop ---
    try:
        result = await run_in_threadpool(pipeline.run, image)
    except SegmentationQualityError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return PredictionResponse(
        label=result.label,
        probability=result.probability,
        threshold=result.threshold,
        is_anemic=result.is_anemic,
        is_mock=result.is_mock,
        warnings=result.warnings,
        crop_preview=_encode_png_data_uri(result.crop_used),
        segmenter_backend=result.segmentation.backend,
        classifier_backend=result.classification.backend,
        disclaimer=settings.disclaimer,
    )


def _encode_png_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
