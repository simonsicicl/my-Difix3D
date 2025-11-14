"""Dataset implementation for DIFIX3D MVP (Step 15.2).

Loads a single scene with images/ and poses.json, resizes to a fixed square size,
and returns tensors ready for model consumption.
"""
from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import warnings

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from .utils_mvp import load_poses_json, sorted_image_paths

class MVPSceneDataset(Dataset):
    """Single-scene dataset returning all views at once.

    Returns a single sample at index 0 with keys:
      - images: Tensor(V, 3, H, W) in [0,1] (H=W=final_size if resizing enabled)
      - poses:  Tensor(V, 4, 4)
    If image_size <= 0 and no downscale provided, original image resolution is preserved.
    """

    def __init__(self, data_root: str, image_size: int = 256, downscale: int | None = None):
        self.data_root = str(Path(data_root).resolve())
        self.image_size = int(image_size)
        self.downscale = int(downscale) if downscale is not None else None
        # Determine final size: if downscale provided use it; else if image_size > 0 use it; else keep original
        if self.downscale is not None:
            self.final_size = self.downscale
        elif self.image_size > 0:
            self.final_size = self.image_size
        else:
            self.final_size = 0  # sentinel meaning: no resizing

        # Load file lists
        self.image_paths = sorted_image_paths(self.data_root)
        self.poses = load_poses_json(self.data_root)
        V_img = len(self.image_paths)
        V_pose = int(self.poses.shape[0])
        assert V_img == V_pose, f"Number of images ({V_img}) != number of poses ({V_pose})"
        if V_img < 2:
            warnings.warn("Less than 2 views; results may be poor.")

        # Preload images into a tensor
        if self.final_size > 0:
            tfm = T.Compose([
                T.Resize((self.final_size, self.final_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ])
        else:
            tfm = T.ToTensor()
        images = []
        for p in self.image_paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
                images.append(tfm(im))
        self.images = torch.stack(images, dim=0)

    def __len__(self) -> int:  # single-scene design
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        assert idx == 0, "MVPSceneDataset only contains a single scene (index 0)."
        return {"images": self.images, "poses": self.poses, "final_size": self.final_size}