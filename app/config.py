"""Application configuration (12-factor, env-overridable).

Every tunable lives here so nothing is hardcoded in the pipeline or API. Override
any field with an env var prefixed ``EYESDEFY_``, e.g.::

    EYESDEFY_CLASSIFIER_BACKEND=convnext_tiny
    EYESDEFY_DEVICE=cpu
    EYESDEFY_CLASSIFIER_THRESHOLD=0.5
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EYESDEFY_",
        env_file=".env",
        extra="ignore",
    )

    # --- Metadata ---
    app_name: str = "Eyes-Defy-Anemia API"
    app_version: str = "0.1.0"

    # --- Model backend selection (mock until the real models are wired) ---
    segmenter_backend: str = "mock"          # mock | aligned_unet (future)
    classifier_backend: str = "convnext_tiny"  # convnext_tiny | mock (mock needs no torch/weights)

    # --- Weights ---
    weights_dir: Path = APP_DIR / "weights"
    classifier_weights: str = "best_convnext_tiny_palpebral_v2_clean.pth"

    # --- Inference config ---
    device: str = "cpu"                    # cpu | cuda
    classifier_input_size: int = 256       # ConvNeXt-Tiny was trained at 256
    classifier_threshold: float = 0.5      # P(anemic) decision threshold

    # --- Mask quality gate (fraction of the frame flagged as tissue) ---
    min_mask_coverage: float = 0.005
    max_mask_coverage: float = 0.60

    # --- Upload validation ---
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")

    # --- CORS (React + Vite dev server defaults) ---
    cors_allow_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    # --- Clinical framing ---
    disclaimer: str = (
        "Research screening tool only. Not a medical device and not a substitute "
        "for clinical diagnosis. Confirm any result with a laboratory hemoglobin test."
    )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so config is parsed once per process."""
    return Settings()
