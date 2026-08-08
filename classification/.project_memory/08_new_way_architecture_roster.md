# `new_way/` — 8-Architecture CNN + Hybrid Roster (classification/, Phase 4)

**Started:** 2026-08-08. Own file per `03_tech_stack_and_rules.md` rule #9.

**Status: registry extended, 16 entry-point scripts + Kaggle notebook built and verified locally. Not yet run on Kaggle.**

---

## 1. Why this exists

Project-author-directed exploration of a fresh architecture roster, alongside (not replacing) the existing 18-combo v2_clean sweep and the Step 1 measurement harness work. Original ask: 3 tiers × 3 families (CNN / Hybrid / ViT) at roughly 15–20M / 70–100M / 120–150+M parameters each. ViT was dropped from the final roster — `vit_b_16`/`vit_l_16` are architectures already fully trained and measured in the v2_clean sweep, so re-running them here would be redundant rather than exploratory.

## 2. Final 8-architecture roster, verified parameter counts

Every count below was measured by actually constructing the model (`weights=None`/`pretrained=False`) and summing `p.numel()` — not recalled from memory or the source paper.

| Family | Tier | Model | Source | Params |
|---|---|---|---|---|
| CNN | Light | EfficientNet-B3 | torchvision | 12.23M (10.70M excl. default 1000-class head) |
| CNN | Light | EfficientNet-B4 | torchvision | 19.34M (17.55M excl. head) |
| CNN | Medium | RegNetY-16GF | torchvision | 83.59M (80.57M excl. head) |
| CNN | Medium | ConvNeXt-Base | torchvision | 88.59M |
| CNN | Heavy | ConvNeXt-Large | torchvision | 197.77M |
| Hybrid | Light | MaxViT-Tiny | torchvision | 30.92M (30.41M excl. head) |
| Hybrid | Medium | MaxViT-Small | **timm** | 68.16M |
| Hybrid | Heavy | CoAtNet-3 | **timm** | 163.64M |

**Two architectures don't exist in torchvision at all** (verified directly, not assumed): torchvision's only MaxViT variant is `maxvit_t`; CoAtNet was never ported to torchvision in any size.

## 3. `timm` adopted as a real project dependency (2026-08-08)

First departure from the standing "torchvision-native only, no `timm`" rule (`03_tech_stack_and_rules.md`, set during the original v2 expansion) — necessary because no torchvision substitute exists for MaxViT-Small or CoAtNet-3 at the requested tiers.

- `timm==1.0.28` added to root `requirements.txt`, along with its two real transitive dependencies `huggingface_hub==1.26.0` and `safetensors==0.8.0`. (It was already incidentally present in the local venv, unpinned and untracked, before this — now formally adopted and recorded.)
- **Verified, not just installed:** both target models built with `pretrained=True`, real weights downloaded from Hugging Face Hub, forward pass run.
  - `maxvit_small_tf_224.in1k` — 68.16M params, output `(1, 768)`. Pretrained tag `in1k` — **standard ImageNet-1k, same regime as every torchvision weight in this project.**
  - `coatnet_3_rw_224.sw_in12k` — 163.64M params, output `(1, 1536)`. Pretrained tag `sw_in12k` — **ImageNet-12k, with no ImageNet-1k fine-tuning stage** (confirmed: no `coatnet_3_rw_224.*_ft_in1k` tag exists in timm's pretrained list, unlike e.g. `coatnet_2_rw_224.sw_in12k_ft_in1k`). **The one architecture in this whole project with a different pretraining regime than everything else** — flag explicitly if reported in the thesis, don't treat as an interchangeable backbone.

## 4. Head-replacement design per architecture — verified by inspecting each model's actual head structure, not assumed

Same "frozen backbone + minimal `Dropout→Linear` trained head" philosophy as every existing architecture in this project, applied per-architecture based on each one's real head structure:

- **EfficientNet-B3/B4** (torchvision): pre-existing hardcoded dropout (p=0.3 / p=0.4) in `classifier[0]` overwritten with the trial's `dropout_rate`, `classifier[1]` replaced with `Linear(in_features, 1)` — identical pattern to EfficientNet-B0/MobileNetV3-Small already in the registry.
- **RegNetY-16GF** (torchvision): bare `model.fc` replaced with `Sequential(Dropout, Linear)` — identical to RegNetY-400MF.
- **ConvNeXt-Base/Large** (torchvision): `classifier[2]` (the final Linear, after the existing `LayerNorm2d`+`Flatten`) replaced with `Sequential(Dropout, Linear)` — identical to ConvNeXt-Tiny.
- **MaxViT-Tiny** (torchvision): its classifier has a **pretrained hidden bottleneck** (`AdaptiveAvgPool2d → Flatten → LayerNorm → Linear(512,512) → Tanh → Linear(512,1000)`) — verified by inspecting the module directly. Only the *final* Linear (`classifier[5]`) is replaced; the hidden 512→512 Linear+Tanh stays frozen as part of the pretrained feature pipeline, so the trainable head stays the same tiny scale (513 params) as every other architecture rather than ~260K if that hidden layer were left trainable.
- **MaxViT-Small / CoAtNet-3** (timm): same finding — both have a `head.pre_logits`-style hidden Linear (768→768 for MaxViT-Small; CoAtNet-3 has none) baked into timm's classifier head class. Built via `timm.create_model(..., num_classes=1)`, then only `head.fc`'s parameters are unfrozen; `head.pre_logits` (where present) stays frozen. `head.drop`'s pre-existing (p=0.0) dropout is overwritten with the trial's `dropout_rate` — same "overwrite existing dropout" pattern already used for MobileNetV3-Small/EfficientNet-B0/B3/B4.

**Verified trainable-parameter counts, all in the same tiny scale as the rest of the project's roster (413–3,025):** efficientnet_b3=1537, efficientnet_b4=1793, regnet_y_16gf=3025, convnext_base=1025, convnext_large=1537, maxvit_t=513, maxvit_small=769, coatnet_3=1537.

## 5. RegNetY-16GF weight-variant decision — verified, not defaulted

Torchvision offers 4 pretrained variants for this architecture: `IMAGENET1K_V1` (80.4% top-1), `IMAGENET1K_V2` (82.9%), `IMAGENET1K_SWAG_E2E_V1` (86.0%), `IMAGENET1K_SWAG_LINEAR_V1` (84.0%). The two higher-accuracy SWAG variants were checked via each weight's own `.transforms()` and require **384×384** (E2E) or **224×224** (LINEAR) input respectively — neither matches this project's uniform 256×256 resize used for every other CNN. `IMAGENET1K_V1` was kept despite being the lowest nominal top-1 of the four, since its own recommended preprocessing (256 resize) is the only variant actually consistent with the existing convention. Same resolution-compatibility reasoning already applied when ViT-L/16 was given `SWAG_LINEAR_V1` instead of the default — there the stronger variant happened to match the existing convention (224 for transformers); here none of the stronger RegNetY variants do, so the default is the consistent choice instead.

## 6. What was built

- **`datapreparepipeline/trainer_engine.py`** — extended, not replaced. Added `import timm` and 8 new `build_*` functions to the same shared `ARCHITECTURE_REGISTRY` every other combo in this project already uses — identical mechanism to how the original 3 architectures became 9 during the v2 expansion. `run_study()`, metrics, plotting: all unchanged, all reused as-is.
- **New `classification/new_way/`** — 16 entry-point scripts (8 architectures × 2 tissue types), generated programmatically from a template (not hand-copy-pasted, same anti-transcription-error precedent as the original 18 `v2_clean_scripts` entries) and spot-checked by reading two in full. Every `model_name` carries a `_new_way` suffix (e.g. `efficientnet_b3_palpebral_new_way`) so results can never collide with any existing `model_name` in the shared `classification/outputs/` (rule #3).
- **New `classification/Kaggle-Notebook/classification-new-way.ipynb`** (31 cells) — Setup/Data cells reused verbatim from the Kaggle-proven `classification-cnn-clean.ipynb`, with one necessary change: the pip-install cell now also installs `timm`. A new "registry sanity check" cell builds and forward-passes all 8 new architectures before any real training starts. 16 training cells, ordered cheapest-first by verified total parameter count (10.70M → 196.23M) — **not** split by CNN/Hybrid family, since unlike the earlier CNN/ViT split, costs interleave here (ConvNeXt-Large at 196M is heavier than any hybrid), so no clean family-based session split exists. `sync_outputs()` after every combo, zips to `new_way_results.zip`.

## 7. Verification actually performed (2026-08-08)

- **Structural check, all 8 architectures × both dropout rates (0.2, 0.5):** built, forward-passed at each architecture's registered resolution, asserted output shape `(B, 1)`, asserted the sampled `dropout_rate` is actually present in a live `Dropout` module. Zero errors — same format as the original 9-architecture verification.
- **Real weight downloads confirmed working** for all 8 (torchvision hub + Hugging Face Hub for the 2 timm models) — not mocked, not `pretrained=False`.
- **Full end-to-end dry run through the real `run_study()` path** (not isolated calls), `MAX_EPOCHS` monkey-patched to 1, `n_trials=1`, two combos chosen to cover both new code branches: a plain torchvision CNN (`regnet_y_16gf`) and a timm-sourced hybrid (`maxvit_small`) — the timm path hadn't been exercised through the full train/eval/Optuna/plot loop before, only a forward pass. Both completed with zero errors: per-country metrics, confusion matrix, ROC curve, `trials.csv`, `study_summary.json`, and all 4 plot types generated correctly for both. Dry-run artifacts (4 logs, 8 plots, 2 checkpoints) deleted afterward — one cleanup-script bug caught and fixed along the way (the checkpoint files are prefixed `best_dryrun_...`, not `dryrun_...`, so the first cleanup pass missed them; found by re-checking rather than trusting the script's own "removed N files" count).
- **Real runtime import test** (not just `py_compile`) confirmed one entry-point script correctly resolves `run_study` through the `../datapreparepipeline` path with the now-17-architecture registry loaded.

## 8. Not yet done

- **Nothing has been run on Kaggle yet.** The notebook is built and locally verified but not launched.
- Not committed or pushed — pending explicit go-ahead, same standing rule as always.
- No `organize_and_compare.py`-equivalent has been run yet (nothing to organize until results exist); the existing script should work unmodified once results land, since combo discovery is dynamic (globs `*_study_summary.json`), not a hardcoded list.
