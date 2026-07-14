# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repository is an early-stage scaffold. The only things present are a Python 3.14.6 virtual environment (`venv/`, no packages installed) and default Claude Code local settings (`.claude/settings.local.json`). There is no source code, dependency manifest, README, or tests yet.

The project name ("EYES-DEFY-ANEMIA") suggests an eye-image-based anemia detection project, but no framework, dependencies, or architecture have been chosen yet.

Update this file once real source code, dependencies, and structure exist.

## Project Progress & Milestones

### Phase 0 — Dataset Standardization (complete)
- Built `scripts/phase0_prepare_dataset.py`: reads `archive.zip` (India + Italy folders) directly, no manual extraction.
- Unified India + Italy metadata into `data/processed/metadata.csv`; labeled anemia via WHO thresholds (Hgb < 13.0 g/dL male, < 12.0 g/dL female; all females assumed non-pregnant — no pregnancy field in source data).
- Extracted raw eye photo + palpebral conjunctiva crop per patient; EXIF-transposed, padded to square (black fill), resized to **256x256**.
- Dropped **`Italy_093`** (ELIMINATO flag found in column G of Italy.xlsx) — not to be confused with `India_093`, which is valid and included.
- Repaired **63/217 palpebral crop PNGs** with a corrupted CRC on the ancillary `iCCP` (embedded color profile) chunk; pixel data (IHDR/PLTE/IDAT/IEND) was intact in all cases, so only the bad ancillary chunk is stripped rather than discarding the image.
- Final dataset: 217 included patients (95 India, 122 Italy) after exclusions.
- **Demographic bias discovered:** female anemia rate is much higher in India than Italy — **India F 87.0% (n=46) vs Italy F 32.5% (n=40)**. (Overall, all-gender rates are India 71.6% vs Italy 18.9% — the 87%/32% figures are the female-only subgroup and should not be quoted as the overall country rate.)
