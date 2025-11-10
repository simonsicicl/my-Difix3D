"""
Model implementation for DIFIX3D MVP

Components:
  - Feature encoder: partial ResNet18 (up to layer1) producing (B*V, C, h, w)
  - View fusion: mean pooling over views -> (B, C, h, w)
  - Minimal UNet-ish core: down -> bottleneck -> upsample -> RGB

Design goals:
  - Keep parameter count low
  - Avoid excessive memory use
  - Return output matching input spatial size (H,W)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Double 3x3 Conv block (classic UNet style) retaining the original class name.

    Structure:
      Conv(in_ch -> out_ch) + BN + ReLU (if act=True)
      Conv(out_ch -> out_ch) + BN + ReLU (if act=True)
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, act: bool = True):
        super().__init__()
        padding = k // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = F.relu(x, inplace=True) if self.act else x
        x = self.bn2(self.conv2(x))
        x = F.relu(x, inplace=True) if self.act else x
        return x


class MVPModel(nn.Module):
    """Multi-view fusion + tiny UNet for reconstruction.

    Forward signature (contract):
      images: Tensor(B, V, 3, H, W)
      poses:  Tensor(V, 4, 4)  (unused so far)
    Returns: Tensor(B, 3, H, W)
    """

    def __init__(self, image_size: int = 256):
        super().__init__()
        self.image_size = image_size
        # UNet-like core
        # Encoder
        enc_channels = 64
        self.enc1 = ConvBlock(3, enc_channels)                    # (H,W)   -> C=64
        self.poo1 = nn.MaxPool2d(2)                               # (H/2,W/2)
        self.enc2 = ConvBlock(enc_channels, enc_channels * 2)     # C=128 at H/2
        self.poo2 = nn.MaxPool2d(2)                               # (H/4,W/4)
        # Bottleneck (128 -> 256)
        self.enc3 = ConvBlock(enc_channels * 2, enc_channels * 4) # C=256 at H/4
        # Decoders
        self.upc1 = nn.ConvTranspose2d(enc_channels * 4, enc_channels * 2, kernel_size=2, stride=2)   # 256 -> 128
        self.dec1 = ConvBlock(enc_channels * 4, enc_channels * 2)                                     # 256 -> 128
        self.upc2 = nn.ConvTranspose2d(enc_channels * 2, enc_channels, kernel_size=2, stride=2)       # 128 -> 64
        self.dec2 = ConvBlock(enc_channels * 2, enc_channels)                                         # 128 -> 64
        # Final output layer
        self.final_up = nn.Conv2d(enc_channels, 3, kernel_size=1)

    def forward(self, images: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = images.shape
        assert C == 3, "Expected 3-channel RGB input"

        # UNet Path (Single pass)
        # Encoding per-view, then fuse across views at two scales
        x = images.view(B * V, C, H, W)
        e1 = self.enc1(x)               # (B*V,64,H,W)
        p1 = self.poo1(e1)              # (B*V,64,H/2,W/2)
        e2 = self.enc2(p1)              # (B*V,128,H/2,W/2)
        p2 = self.poo2(e2)              # (B*V,128,H/4,W/4)
        # Middle
        bt = self.enc3(p2)               # (B*V,256,H/4,W/4)
        # Decoding
        u1 = self.upc1(bt)                  # (B*V,128,H/2,W/2)
        cat1 = torch.cat([u1, e2], dim=1)   # (B*V,256,H/2,W/2)
        d1 = self.dec1(cat1)                # (B*V,128,H/2,W/2)
        u2 = self.upc2(d1)                  # (B*V,64,H,W)
        cat2 = torch.cat([u2, e1], dim=1)   # (B*V,128,H,W)
        d2 = self.dec2(cat2)                # (B*V,64,H,W)

        # Upsample to original size (H,W) if needed
        # scale_needed_h = H / d2.shape[-2]
        # scale_needed_w = W / d2.shape[-1]
        # if scale_needed_h != 1 or scale_needed_w != 1:
        #     print(f"Upsampling from ({d2.shape[-2]}, {d2.shape[-1]}) to ({H}, {W})")
        #     d2 = F.interpolate(d2, size=(H, W), mode="bilinear", align_corners=False)

        out = self.final_up(d2)      # (B,3,H,W)
        return out.clamp(0.0, 1.0)
