"""
Phase 2: Standard U-Net (Ronneberger et al., 2015) for palpebral conjunctiva
segmentation.

Input:  3-channel RGB image, 256x256.
Output: 1-channel map of raw logits (no Sigmoid). Raw logits are paired with
torch.nn.BCEWithLogitsLoss, which applies a numerically-stable sigmoid
internally -- this avoids the vanishing-gradient/precision issues of a
separate Sigmoid + BCELoss. Apply torch.sigmoid(output) explicitly when you
need an actual probability/mask at inference time.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) x2, same-padding so spatial size is unchanged."""

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


class Down(nn.Module):
    """One contracting-path step: 2x2 max pool, then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """One expanding-path step: upsample, concatenate the matching skip
    connection, then DoubleConv. Supports either learned transposed
    convolution or fixed bilinear upsampling."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in_channels = in_channels + out_channels
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            conv_in_channels = out_channels * 2
        self.conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Symmetrical encoder-decoder with skip connections. With the default
    base_channels=64 and a 256x256 input, the bottleneck is 1024 channels
    at 16x16 (four 2x downsamples: 256 -> 128 -> 64 -> 32 -> 16)."""

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

        self.up1 = Up(base_channels * 16, base_channels * 8, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)  # base_channels    @ H,    W
        x2 = self.down1(x1)   # base_channels*2  @ H/2,  W/2
        x3 = self.down2(x2)   # base_channels*4  @ H/4,  W/4
        x4 = self.down3(x3)   # base_channels*8  @ H/8,  W/8
        x5 = self.down4(x4)   # base_channels*16 @ H/16, W/16 (bottleneck)

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

    model = UNet(in_channels=3, out_channels=1).to(device)

    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    output = model(dummy_input)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype} | device: {output.device}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
