"""
TransUNet (Chen et al., 2021) -- the "strong" (100-200M) Hybrid entry in the
9-architecture pretrained segmentation sweep (CLAUDE.md Sec 1.4 follow-up).

Implementation note (deliberate deviation from "vendor the original repo"):
there is no well-maintained PyTorch package for this architecture -- the
`transunet` PyPI package has only 3 releases with unverified provenance, and
the original Beckschen/TransUNet research repo mixes an old TF checkpoint-
conversion step into its PyTorch code, making a clean vendor-and-adapt a
higher-risk path than this project's "verify empirically" rule is
comfortable with. Instead, this is a faithful compositional rebuild from
two independently pretrained, actively-maintained `timm` components:

1. A `timm` ResNet50 (`features_only=True`, ImageNet-pretrained) supplies
   the multi-scale CNN feature maps -- both the deepest one (bottleneck
   input to the transformer stage) and the four shallower ones (skip
   connections into the decoder), exactly the same two roles skip
   connections play in this project's own hand-built unet.py.
2. A `timm` ViT-B/16 (ImageNet-pretrained)'s 12 transformer encoder blocks
   + final LayerNorm are reused for global context modeling over the
   ResNet's bottleneck feature map -- NOT its patch_embed (the input here
   is a CNN feature map, not raw image patches, so patch_embed does not
   apply) or its classification head (discarded). The bottleneck feature
   map is projected to the ViT's hidden dim via a fresh 1x1 conv, flattened
   into a token sequence, and given a fresh, bilinearly-interpolated copy
   of the ViT's own pretrained positional embedding (interpolated from its
   native 14x14 grid to whatever grid size the bottleneck feature map
   actually is at this project's 256x256 input) -- the same interpolation
   trick commonly used to fine-tune ViT at a non-native resolution.

This keeps every pretrained-vs-scratch claim honest and inspectable in code
(unlike a vendored repo, where it's hard to audit exactly what loaded and
what didn't) while reproducing TransUNet's actual architectural idea: CNN
for local detail + skip connections, transformer for global context at the
bottleneck, CNN decoder ("CUP" in the original paper) to get back to full
resolution.

Contract matches every other model in this project: forward(x) with
x: [B, 3, H, W] -> raw logits [B, 1, H, W] (no Sigmoid applied internally).
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Same block shape as models/segmentation/unet.py's DoubleConv, kept as
    an independent copy here (not imported) since transunet.py is meant to
    be a self-contained pretrained-model file -- see the module docstring's
    reasoning for keeping this file's provenance easy to audit in one place."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderUpBlock(nn.Module):
    """One CUP (cascaded upsampler) stage: bilinear upsample x2, concatenate
    with the matching ResNet skip feature (if any -- the final stage has
    none, since the ResNet stem's shallowest feature is already at stride 2,
    not stride 1), then a DoubleConv to fuse."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = _DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        vit_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
    ):
        super().__init__()

        # --- CNN encoder: multi-scale skip features + bottleneck input ---
        # out_indices=(0,1,2,3,4) on a timm resnet50 gives strides
        # (2,4,8,16,32) with channels (64,256,512,1024,2048) for a 256x256
        # input -> spatial (128,64,32,16,8).
        self.cnn_encoder = timm.create_model(
            "resnet50", features_only=True, pretrained=pretrained, in_chans=in_channels,
            out_indices=(0, 1, 2, 3, 4),
        )
        encoder_channels = self.cnn_encoder.feature_info.channels()  # [64, 256, 512, 1024, 2048]

        # --- Transformer bottleneck: reuse a pretrained ViT's blocks only ---
        vit = timm.create_model(vit_name, pretrained=pretrained)
        self.vit_hidden_dim = vit.embed_dim
        self.vit_native_grid = vit.patch_embed.grid_size  # e.g. (14, 14) for vit_base_patch16_224
        self.vit_pos_embed = vit.pos_embed  # [1, 1 + 14*14, hidden_dim] (includes the CLS token slot)
        self.vit_has_cls_token = getattr(vit, "cls_token", None) is not None
        self.vit_blocks = vit.blocks
        self.vit_norm = vit.norm
        del vit  # drop patch_embed/head and everything else we don't use

        self.bottleneck_proj = nn.Conv2d(encoder_channels[-1], self.vit_hidden_dim, kernel_size=1)
        self.bottleneck_unproj = nn.Conv2d(self.vit_hidden_dim, encoder_channels[-1], kernel_size=1)

        # --- CUP decoder: bottleneck -> stride16 -> 8 -> 4 -> 2 -> full res ---
        c4, c3, c2, c1, c0 = encoder_channels[::-1]  # 2048, 1024, 512, 256, 64
        decoder_channels = [256, 128, 64, 32, 16]
        self.up1 = _DecoderUpBlock(c4, c3, decoder_channels[0])   # stride32 -> 16, concat skip @1024ch
        self.up2 = _DecoderUpBlock(decoder_channels[0], c2, decoder_channels[1])  # 16 -> 8, concat @512ch
        self.up3 = _DecoderUpBlock(decoder_channels[1], c1, decoder_channels[2])  # 8 -> 4, concat @256ch
        self.up4 = _DecoderUpBlock(decoder_channels[2], c0, decoder_channels[3])  # 4 -> 2, concat @64ch
        self.up5 = _DecoderUpBlock(decoder_channels[3], 0, decoder_channels[4])   # 2 -> 1 (full res), no skip

        self.out_conv = nn.Conv2d(decoder_channels[4], out_channels, kernel_size=1)

    def _interpolated_pos_embed(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Bilinearly resizes the pretrained ViT's positional embedding grid
        from its native size (e.g. 14x14 for a 224/16 ViT-B/16) to whatever
        grid size the ResNet's bottleneck feature map actually has at this
        project's input resolution -- the standard trick for reusing a
        ViT's pretrained position embeddings at a non-native token count."""
        pos_embed = self.vit_pos_embed
        n_prefix = 1 if self.vit_has_cls_token else 0  # drop CLS-token slot, unused here
        patch_pos_embed = pos_embed[:, n_prefix:, :]

        native_h, native_w = self.vit_native_grid
        patch_pos_embed = patch_pos_embed.reshape(1, native_h, native_w, self.vit_hidden_dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(grid_h, grid_w), mode="bicubic", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.vit_hidden_dim)
        return patch_pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn_encoder(x)  # 5 feature maps, shallow -> deep
        skip1, skip2, skip3, skip4, bottleneck = features

        b, c, h, w = bottleneck.shape
        tokens = self.bottleneck_proj(bottleneck).flatten(2).transpose(1, 2)  # [B, h*w, hidden_dim]
        tokens = tokens + self._interpolated_pos_embed(h, w).to(tokens.dtype)

        for block in self.vit_blocks:
            tokens = block(tokens)
        tokens = self.vit_norm(tokens)

        transformed = tokens.transpose(1, 2).reshape(b, self.vit_hidden_dim, h, w)
        transformed = self.bottleneck_unproj(transformed)

        d = self.up1(transformed, skip4)
        d = self.up2(d, skip3)
        d = self.up3(d, skip2)
        d = self.up4(d, skip1)
        d = self.up5(d, None)

        if d.shape[-2:] != x.shape[-2:]:
            d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return self.out_conv(d)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransUNet(in_channels=3, out_channels=1).to(device)
    x = torch.randn(2, 3, 256, 256, device=device)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(y.shape)}, dtype: {y.dtype}, device: {y.device}")
    print(f"Parameter count: {n_params:,}")
