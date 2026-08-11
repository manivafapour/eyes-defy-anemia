# Deploying Eyes-Defy-Anemia to Hugging Face Spaces

One Docker container: FastAPI serves the React SPA **and** the API on a single port.
CPU-only — no GPU required.

## What gets deployed
- `Dockerfile` — multi-stage (builds the React app, then a slim Python runtime)
- `app/` — FastAPI code + frontend source + the ConvNeXt-Tiny checkpoint (`app/weights/*.pth`)
- `README.md` — the HF Space config (from `deploy/README_hf.md`)

---

## Option A — test the image locally first (recommended; needs Docker Desktop)

From the repo root (`D:\khaje\EYES-DEFY-ANEMIA`):

```bash
docker build -t eyesdefy .
```

```bash
docker run --rm -p 7860:7860 eyesdefy
```

Open **http://localhost:7860**. The first build downloads Node + Python + CPU torch
(~5–10 min); later builds are cached.

---

## Option B — deploy to Hugging Face Spaces

Prereqs: a free HF account, `git`, and `git-lfs` (https://git-lfs.com).

**1. Create the Space.** Go to https://huggingface.co/new-space → **SDK: Docker** →
**Blank** → name it `eyes-defy-anemia`.

**2. Clone it** (run these in a folder *outside* your project):

```bash
git clone https://huggingface.co/spaces/<your-username>/eyes-defy-anemia
```

**3. Copy the deployment files in** (PowerShell, from the cloned Space folder):

```powershell
copy D:\khaje\EYES-DEFY-ANEMIA\Dockerfile .
copy D:\khaje\EYES-DEFY-ANEMIA\.dockerignore .
copy D:\khaje\EYES-DEFY-ANEMIA\deploy\README_hf.md README.md
robocopy D:\khaje\EYES-DEFY-ANEMIA\app .\app /E /XD node_modules dist __pycache__ .vite
del app\.gitignore   # so the checkpoint can be committed; caches already excluded by robocopy
```

`robocopy` brings the whole `app/` — code, frontend source, and `weights\*.pth` — while
skipping `node_modules`, `dist`, and caches.

**4. Track the checkpoint with LFS** (it is ~107 MB):

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

**5. Commit and push:**

```bash
git add -A
git commit -m "Deploy Eyes-Defy-Anemia demo"
git push
```

Confirm the checkpoint went through LFS before/after pushing:

```bash
git lfs ls-files
```

**6. Watch it build.** Open your Space page → **Logs**. First build is ~5–10 min (torch
install). When it reports the app is running, the demo is live at
`https://huggingface.co/spaces/<your-username>/eyes-defy-anemia`.

---

## Notes
- **Stage 1 is still mocked** — the deployed app runs the *real* classifier on a
  placeholder crop and says so in the UI. Swap in the aligned segmenter later, rebuild
  `app/frontend` if changed, and push again to redeploy.
- **Runtime config** via the Space's **Settings → Variables**: `EYESDEFY_CLASSIFIER_THRESHOLD`,
  `EYESDEFY_CLASSIFIER_BACKEND` (`convnext_tiny` or `mock`), `EYESDEFY_DEVICE`.
- **Reproducibility:** the container installs the latest CPU `torch`; predictions match the
  training run (verified by `app/tests/test_convnext_parity.py`). Pin `torch==<version>` in
  the Dockerfile if you want to lock it exactly.
