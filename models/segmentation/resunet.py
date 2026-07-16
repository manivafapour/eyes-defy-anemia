"""
Phase 2, Model 3: ResUNet (Zhang et al., 2017, "Road Extraction by Deep
Residual U-Net") for palpebral conjunctiva segmentation.

Same encoder/decoder topology and I/O contract as the standard U-Net
(models/segmentation/unet.py): 3x256x256 RGB in, 1x256x256 raw logits out
(no Sigmoid -- pair with BCEWithLogitsLoss). The difference is the core
building block: every DoubleConv is replaced by a residual block (two 3x3
Conv-BN layers plus an identity/projection shortcut added before the final
ReLU), which is intended to ease gradient flow through the network and let
each block learn a residual refinement rather than a full transformation.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3x3 Conv-BN layers with a shortcut connection added before the
    final activation. The shortcut is a 1x1 Conv-BN projection when the
    channel count changes (every block in this network, since each stage
    changes channel depth), otherwise an identity."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResDown(nn.Module):
    """One contracting-path step: 2x2 max pool, then a ResidualBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResUp(nn.Module):
    """One expanding-path step: upsample, concatenate the matching skip
    connection, then a ResidualBlock (in place of DoubleConv)."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in_channels = in_channels + out_channels
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            conv_in_channels = out_channels * 2
        self.conv = ResidualBlock(conv_in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResUNet(nn.Module):
    """Same channel progression and spatial reduction as UNet/AttentionUNet
    (64 -> 128 -> 256 -> 512 -> 1024 bottleneck, 256x256 -> 16x16), with
    ResidualBlock in place of DoubleConv throughout."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = False,
    ):
        super().__init__()

        self.in_conv = ResidualBlock(in_channels, base_channels)
        self.down1 = ResDown(base_channels, base_channels * 2)
        self.down2 = ResDown(base_channels * 2, base_channels * 4)
        self.down3 = ResDown(base_channels * 4, base_channels * 8)
        self.down4 = ResDown(base_channels * 8, base_channels * 16)

        self.up1 = ResUp(base_channels * 16, base_channels * 8, bilinear)
        self.up2 = ResUp(base_channels * 8, base_channels * 4, bilinear)
        self.up3 = ResUp(base_channels * 4, base_channels * 2, bilinear)
        self.up4 = ResUp(base_channels * 2, base_channels, bilinear)

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

    model = ResUNet(in_channels=3, out_channels=1).to(device)

    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    output = model(dummy_input)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype} | device: {output.device}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
