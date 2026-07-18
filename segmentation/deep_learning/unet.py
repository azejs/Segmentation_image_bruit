"""
U-Net architecture for semantic segmentation.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015.

Architecture:
    Encoder (contracting path): 4 × [Conv-BN-ReLU × 2 + MaxPool]
    Bottleneck:                  Conv-BN-ReLU × 2
    Decoder (expanding path):    4 × [UpConv + skip-concat + Conv-BN-ReLU × 2]
    Output head:                 1×1 Conv → binary/multi-class mask
"""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Downsampling block: MaxPool2d + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    """
    Upsampling block: bilinear up-sampling + skip-connection concatenation
    + DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Crop skip connection if spatial sizes differ (input not divisible by 16)
        if x.shape != skip.shape:
            skip = skip[:, :, : x.shape[2], : x.shape[3]]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary or multi-class segmentation.

    Args:
        in_channels:  Number of input channels (1=gray, 3=RGB).
        out_channels: Number of output classes (1 for binary segmentation).
        features:     Channel widths for the four encoder levels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Encoder
        self.enc1 = _DoubleConv(in_channels, features[0])
        self.enc2 = _Down(features[0], features[1])
        self.enc3 = _Down(features[1], features[2])
        self.enc4 = _Down(features[2], features[3])

        # Bottleneck
        self.bottleneck = _Down(features[3], features[3] * 2)

        # Decoder
        self.dec4 = _Up(features[3] * 2 + features[3], features[3])
        self.dec3 = _Up(features[3] + features[2],      features[2])
        self.dec2 = _Up(features[2] + features[1],      features[1])
        self.dec1 = _Up(features[1] + features[0],      features[0])

        # Output
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path (save skip connections)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder path
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)
