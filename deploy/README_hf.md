---
title: Eyes Defy Anemia
emoji: 👁️
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Eyes-Defy-Anemia — Anemia Screening Demo

Two-stage pipeline: an eye photo → **(Stage 1)** conjunctiva segmentation →
**(Stage 2)** a ConvNeXt-Tiny anemia classifier → *Anemic / Non-Anemic*.

**Demo status.** Stage 1 is currently **mocked** (a placeholder crop); the ConvNeXt-Tiny
classifier is real (validation F1 ≈ 0.93). This is a **research screening demo, not a
medical device** — every result carries that disclaimer, and the app says so in the UI.

Built with FastAPI (serving both the API and a React + Vite SPA) in a single CPU container.
