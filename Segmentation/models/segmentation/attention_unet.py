"""
Phase 2, Model 2: Attention U-Net (Oktay et al., 2018) for palpebral
conjunctiva segmentation.

Same encoder and I/O contract as the standard U-Net (models/segmentation/
unet.py): 3x256x256 RGB in, 1x256x256 raw logits out (no Sigmoid --
pair with BCEWithLogitsLoss). The difference is in the decoder: each skip
connection passes through an additive attention gate, conditioned on the
decoder's own (already-upsampled) gating signal, before concatenation. The
gate learns a per-pixel relevance weight in [0, 1] that suppresses
background/irrelevant activations in the skip connection -- intended to
help the model focus on the (often small, irregularly-shaped) conjunctiva
region rather than the padding/background that dominates most of the crop.
"""

import torch
import torch.nn as nn

try:
    from .unet import DoubleConv, Down
except ImportError:  # running as a standalone script, not as a package member
    from unet import DoubleConv, Down


class AttentionGate(nn.Module):
    """Additive attention gate. Projects the gating signal `g` and the skip
    connection `x` (same spatial resolution, since `g` here is taken after
    the decoder's own upsampling) into a shared intermediate space, sums
    them, and squashes through a 1x1 conv + Sigmoid to get a per-pixel
    attention coefficient in [0, 1], which then rescales `x`."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        attention = self.relu(self.gate_conv(g) + self.skip_conv(x))
        attention = self.psi(attention)
        return x * attention


class AttentionUp(nn.Module):
    """Upsample, gate the skip connection through an AttentionGate, then
    concatenate and DoubleConv -- the attention-U-Net analogue of unet.Up."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            gate_channels = in_channels
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            gate_channels = out_channels

        self.attention_gate = AttentionGate(
            gate_channels=gate_channels,
            skip_channels=out_channels,
            inter_channels=max(out_channels // 2, 1),
        )
        conv_in_channels = (gate_channels + out_channels) if bilinear else (out_channels * 2)
        self.conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        skip = self.attention_gate(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """Standard U-Net encoder (reused from unet.py) + attention-gated decoder.
    Same channel progression as UNet: 64 -> 128 -> 256 -> 512 -> 1024 bottleneck,
    same 256x256 -> 16x16 spatial reduction over 4 downsampling stages."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = False,
    ):
        super().__init__()

        self.in_conv = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = AttentionUp(base_channels * 16, base_channels * 8, bilinear)
        self.up2 = AttentionUp(base_channels * 8, base_channels * 4, bilinear)
        self.up3 = AttentionUp(base_channels * 4, base_channels * 2, bilinear)
        self.up4 = AttentionUp(base_channels * 2, base_channels, bilinear)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)  # raw logits, [B, out_channels, H, W]


# --------------------------------------------------------------------------
# Verification block
# --------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AttentionUNet(in_channels=3, out_channels=1).to(device)

    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    output = model(dummy_input)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype} | device: {output.device}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
