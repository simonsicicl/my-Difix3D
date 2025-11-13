"""
Defect synthesis utilities for DIFIX-style training.

Applies artificial corruptions to clean RGB tensors in [0,1].

API:
  corrupt_batch(images, kinds=(...), prob=0.7, strength=1.0) -> images_corrupt

Supported kinds:
  - "mask": random rectangular cutouts/holes
  - "noise": gaussian noise
  - "blur": gaussian blur
  - "jpeg": jpeg compression artifacts
  - "occlusion": solid color blocks
  - "color": color jitter (brightness/contrast/saturation)

Notes:
  - All ops preserve tensor range [0,1].
  - Input shape can be (B,V,3,H,W) or (B,3,H,W).
"""
from __future__ import annotations

from typing import Iterable, List
import io
import random

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:  # (B,V,3,H,W) -> (B*V,3,H,W)
        b, v, c, h, w = x.shape
        return x.reshape(b * v, c, h, w)
    elif x.dim() == 4:
        return x
    else:
        raise ValueError(f"Unsupported shape: {tuple(x.shape)}")


def _restore_shape(x_flat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if ref.dim() == 5:
        b, v = ref.shape[:2]
        c, h, w = x_flat.shape[1:]
        return x_flat.reshape(b, v, c, h, w)
    return x_flat


def _rand_rect_mask(h: int, w: int, n: int = 3, scale: float = 0.25) -> torch.Tensor:
    m = torch.zeros((h, w), dtype=torch.float32)
    for _ in range(n):
        rh = max(1, int(random.uniform(0.05, scale) * h))
        rw = max(1, int(random.uniform(0.05, scale) * w))
        y0 = random.randint(0, max(0, h - rh))
        x0 = random.randint(0, max(0, w - rw))
        m[y0:y0+rh, x0:x0+rw] = 1.0
    return m


def defect_mask(x: torch.Tensor, n: int = 3, scale: float = 0.3) -> torch.Tensor:
    b, c, h, w = x.shape
    masks = []
    for _ in range(b):
        m = _rand_rect_mask(h, w, n=n, scale=scale)
        masks.append(m)
    m = torch.stack(masks, 0).unsqueeze(1)  # (B,1,H,W)
    return m.to(x.device)


def apply_mask(x: torch.Tensor, mode: str = "zeros", n: int = 3, scale: float = 0.3) -> torch.Tensor:
    m = defect_mask(x, n=n, scale=scale)
    if mode == "zeros":
        return x * (1 - m)
    elif mode == "mean":
        mean = x.mean(dim=(2, 3), keepdim=True)
        return x * (1 - m) + mean * m
    else:
        return x * (1 - m)


def apply_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(0.0, 1.0)


def apply_blur(x: torch.Tensor, k: int = 5, sigma: float = 1.0) -> torch.Tensor:
    # separable gaussian via conv
    if k % 2 == 0:
        k += 1
    half = k // 2
    t = torch.arange(-half, half + 1, device=x.device).float()
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = g / g.sum()
    g_col = g.view(1, 1, k, 1)
    g_row = g.view(1, 1, 1, k)
    c = x.shape[1]
    x = F.conv2d(x, g_col.expand(c, 1, k, 1), padding=(half, 0), groups=c)
    x = F.conv2d(x, g_row.expand(c, 1, 1, k), padding=(0, half), groups=c)
    return x.clamp(0.0, 1.0)


def apply_jpeg(x: torch.Tensor, quality: int = 30) -> torch.Tensor:
    x_cpu = (x.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    out = []
    for img in x_cpu:
        pil = Image.fromarray(img.permute(1, 2, 0).numpy())
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil2 = Image.open(buf).convert("RGB")
        t = TF.to_tensor(pil2)
        out.append(t)
    y = torch.stack(out, 0).to(x.device)
    return y.clamp(0.0, 1.0)


def apply_occlusion(x: torch.Tensor, n: int = 2, scale: float = 0.2) -> torch.Tensor:
    b, c, h, w = x.shape
    out = x.clone()
    for i in range(b):
        for _ in range(n):
            rh = max(1, int(random.uniform(0.05, scale) * h))
            rw = max(1, int(random.uniform(0.05, scale) * w))
            y0 = random.randint(0, max(0, h - rh))
            x0 = random.randint(0, max(0, w - rw))
            color = torch.rand((c, 1, 1), device=x.device)
            out[i, :, y0:y0+rh, x0:x0+rw] = color
    return out


def apply_color_jitter(x: torch.Tensor, b: float = 0.2, c_: float = 0.2, s: float = 0.2) -> torch.Tensor:
    out = []
    for img in x:
        pil = TF.to_pil_image(img)
        img2 = TF.adjust_brightness(pil, 1.0 + random.uniform(-b, b))
        img2 = TF.adjust_contrast(img2, 1.0 + random.uniform(-c_, c_))
        img2 = TF.adjust_saturation(img2, 1.0 + random.uniform(-s, s))
        out.append(TF.to_tensor(img2))
    return torch.stack(out, 0).to(x.device).clamp(0.0, 1.0)


def corrupt_batch(images: torch.Tensor, kinds: Iterable[str], prob: float = 0.7, strength: float = 1.0) -> torch.Tensor:
    x = _ensure_bchw(images)
    x_cor = x.clone()
    kinds = list(kinds)
    if not kinds:
        return _restore_shape(x_cor, images)

    if random.random() < prob and "mask" in kinds:
        x_cor = apply_mask(x_cor, n=int(2 + 3 * strength), scale=0.15 + 0.25 * strength)
    if random.random() < prob and "noise" in kinds:
        x_cor = apply_noise(x_cor, sigma=0.05 * strength)
    if random.random() < prob and "blur" in kinds:
        x_cor = apply_blur(x_cor, k=5, sigma=1.0 + 1.5 * strength)
    if random.random() < prob and "jpeg" in kinds:
        q = max(5, int(60 - 40 * strength))
        x_cor = apply_jpeg(x_cor, quality=q)
    if random.random() < prob and "occlusion" in kinds:
        x_cor = apply_occlusion(x_cor, n=int(1 + 2 * strength), scale=0.15 + 0.25 * strength)
    if random.random() < prob and "color" in kinds:
        x_cor = apply_color_jitter(x_cor, b=0.2 * strength, c_=0.2 * strength, s=0.2 * strength)

    return _restore_shape(x_cor, images)


__all__ = [
    "corrupt_batch",
    "apply_mask",
    "apply_noise",
    "apply_blur",
    "apply_jpeg",
    "apply_occlusion",
    "apply_color_jitter",
]
