"""
ARCHITECTURE_REGISTRY for the 9-model pretrained segmentation sweep -- 3 CNN
+ 3 Hybrid (CNN+Transformer) + 3 Pure Transformer, at increasing parameter
tiers, all pretrained and fine-tuned (CLAUDE.md Sec 1.4 follow-up: expanding
segmentation beyond the original 3 hand-built, from-scratch U-Net variants).

Kept in its own file, separate from unet.py/attention_unet.py/resunet.py, so
those 3 models' import surface stays free of the new heavy dependencies this
file needs (segmentation-models-pytorch, timm, transformers, and the
sibling transunet.py).

Every build_fn is a ZERO-ARG callable returning a freshly-constructed
nn.Module satisfying this project's one hard contract: forward(x) with
x: [B, 3, H, W] -> raw logits [B, 1, H, W] (H=W=input_size, no Sigmoid
applied internally) -- see trainer_engine.py's `build_model` parameter,
called once per Optuna trial exactly like `model_cls(in_channels=3,
out_channels=1)` is for the 3 original models.

Measured (not assumed) parameter counts and forward-pass shapes, confirmed
2026-08-08 via this file's own __main__ self-test block (CNN/Hybrid
entries) and standalone checks for the 3 HF-based Transformer entries and
transunet.py's own self-test (both a forward pass and a real 1-trial/
1-epoch dry run through trainer_engine.run_study() were used to confirm
these, not just a bare construction):
  cnn_base_efficientnet_b1_unet             8,757,105   (target 8-12M)
  cnn_mid_resnet101_deeplabv3plus          45,669,713   (target 40-50M)
  cnn_strong_convnext_large_unet          203,268,113   (target 100-200M, slightly over)
  hybrid_base_coatnet0_unet                30,793,619   (target 20-30M, slightly over)
  hybrid_mid_coatnet2_unet                 77,965,273   (target 50-70M, slightly over)
  hybrid_strong_transunet                 120,876,817   (target 100-200M)
  transformer_base_segformer_b2            27,347,393   (target 25-50M)
  transformer_mid_swin_base_upernet       121,302,394   (target 100-120M)
  transformer_strong_swin_large_upernet   233,851,318   (target 200M+)

"Pretrained" means different things per family, stated honestly rather than
implied uniform (CLAUDE.md's own convention for this kind of caveat):
CNN/Hybrid entries (except TransUNet, which mixes both) have a pretrained
ENCODER ONLY -- the decoder trains from scratch, same as this project's
original 3 hand-built models' decoders always have. The 3 Transformer
entries have a pretrained ENCODER + DECODER (fine-tuned from an ADE20K
semantic-segmentation checkpoint), with only the final per-class classifier
layer reinitialized for this project's single-class binary task.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, UperNetForSemanticSegmentation

try:
    from .transunet import TransUNet
except ImportError:
    from transunet import TransUNet


# --------------------------------------------------------------------------
# HuggingFace adapter -- makes SegFormer/UperNet satisfy the raw-logits
# [B, 1, H, W] contract every other model in this project already has.
# --------------------------------------------------------------------------
class _HFSegmentationAdapter(nn.Module):
    """HF semantic-segmentation heads commonly output logits at a reduced
    stride relative to the input (e.g. SegFormer-B2 at 512x512 input ->
    128x128 logits, stride 4) -- this wraps a HF model and upsamples its
    `.logits` output back to the input's own spatial size, so it drops into
    the existing sigmoid-threshold-then-Dice/IoU pipeline (trainer_engine.py
    evaluate()) unchanged, the same way every hand-built model here already
    outputs at full resolution directly."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.hf_model(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


# --------------------------------------------------------------------------
# Builders -- one zero-arg factory per architecture. Each constructs a FRESH
# model (re-downloading nothing after the first call, since huggingface_hub/
# timm both cache weights locally, but re-initializing the reinitialized
# final-layer weights fresh) -- required since trainer_engine.py calls
# build_fn() once per Optuna trial, same as model_cls(...) is for the
# original 3 models.
# --------------------------------------------------------------------------
def _build_unet_efficientnet_b1() -> nn.Module:
    return smp.Unet(encoder_name="efficientnet-b1", encoder_weights="imagenet", in_channels=3, classes=1)


def _build_deeplabv3plus_resnet101() -> nn.Module:
    return smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=1)


def _build_unet_convnext_large() -> nn.Module:
    return smp.Unet(encoder_name="tu-convnext_large", encoder_weights="imagenet", in_channels=3, classes=1)


def _build_unet_coatnet_0() -> nn.Module:
    return smp.Unet(encoder_name="tu-coatnet_0_rw_224", encoder_weights="imagenet", in_channels=3, classes=1)


def _build_unet_coatnet_2() -> nn.Module:
    return smp.Unet(encoder_name="tu-coatnet_2_rw_224", encoder_weights="imagenet", in_channels=3, classes=1)


def _build_transunet() -> nn.Module:
    return TransUNet(in_channels=3, out_channels=1, pretrained=True)


def _build_segformer_b2() -> nn.Module:
    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512", num_labels=1, ignore_mismatched_sizes=True,
    )
    return _HFSegmentationAdapter(hf_model)


def _build_upernet_swin_base() -> nn.Module:
    hf_model = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-swin-base", num_labels=1, ignore_mismatched_sizes=True,
    )
    return _HFSegmentationAdapter(hf_model)


def _build_upernet_swin_large() -> nn.Module:
    hf_model = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-swin-large", num_labels=1, ignore_mismatched_sizes=True,
    )
    return _HFSegmentationAdapter(hf_model)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
ARCHITECTURE_REGISTRY = {
    "cnn_base_efficientnet_b1_unet": {
        "build_fn": _build_unet_efficientnet_b1, "input_size": 256, "family": "cnn", "tier": "base",
    },
    "cnn_mid_resnet101_deeplabv3plus": {
        "build_fn": _build_deeplabv3plus_resnet101, "input_size": 256, "family": "cnn", "tier": "mid",
    },
    "cnn_strong_convnext_large_unet": {
        "build_fn": _build_unet_convnext_large, "input_size": 256, "family": "cnn", "tier": "strong",
    },
    "hybrid_base_coatnet0_unet": {
        "build_fn": _build_unet_coatnet_0, "input_size": 224, "family": "hybrid", "tier": "base",
    },
    "hybrid_mid_coatnet2_unet": {
        "build_fn": _build_unet_coatnet_2, "input_size": 224, "family": "hybrid", "tier": "mid",
    },
    "hybrid_strong_transunet": {
        "build_fn": _build_transunet, "input_size": 256, "family": "hybrid", "tier": "strong",
    },
    "transformer_base_segformer_b2": {
        "build_fn": _build_segformer_b2, "input_size": 512, "family": "transformer", "tier": "base",
    },
    "transformer_mid_swin_base_upernet": {
        "build_fn": _build_upernet_swin_base, "input_size": 512, "family": "transformer", "tier": "mid",
    },
    "transformer_strong_swin_large_upernet": {
        "build_fn": _build_upernet_swin_large, "input_size": 512, "family": "transformer", "tier": "strong",
    },
}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    for name, entry in ARCHITECTURE_REGISTRY.items():
        size = entry["input_size"]
        model = entry["build_fn"]().to(device)
        model.eval()
        x = torch.randn(2, 3, size, size, device=device)
        with torch.no_grad():
            y = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"{name:42s} family={entry['family']:11s} tier={entry['tier']:6s} "
            f"input_size={size:4d} out_shape={tuple(y.shape)} params={n_params:,}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
