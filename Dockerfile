# syntax=docker/dockerfile:1
# Single-container build for the Eyes-Defy-Anemia serving app.
# Stage 1 builds the React SPA; stage 2 is a slim CPU Python runtime that serves
# both the built SPA (at /) and the FastAPI API (at /api) on one port.

# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend
WORKDIR /build
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build          # -> /build/dist

# ---- Stage 2: Python runtime ----
FROM python:3.11-slim AS runtime

# Shared libs that opencv (pulled in by albumentations) needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps as root into system site-packages (world-readable; uvicorn lands on
# the default PATH). CPU-only torch first from its wheel index, so the torch/
# torchvision pins in requirements.txt are already satisfied and get skipped.
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Drop to a non-root user for runtime (Hugging Face Spaces best practice, UID 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    EYESDEFY_CLASSIFIER_BACKEND=convnext_tiny \
    EYESDEFY_DEVICE=cpu \
    PORT=7860
WORKDIR /home/user

# App code (includes the checkpoint in app/weights/) + the built SPA from stage 1.
COPY --chown=user app/ ./app/
COPY --chown=user --from=frontend /build/dist ./app/frontend/dist

EXPOSE 7860
# ${PORT:-7860} keeps it portable: HF Spaces uses app_port 7860; Render/Railway inject PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
