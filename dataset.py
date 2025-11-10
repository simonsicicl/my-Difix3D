"""
Dataset implementation for DIFIX3D MVP

Loads a single scene with images/ and poses.json, resizes to a fixed square size,
and returns tensors ready for model consumption.
"""
import json
import warnings
from typing import Dict, Any, List
from pathlib import Path
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image


class MVPSceneDataset(Dataset):
    """Single-scene dataset returning all views at once.

    Returns a single sample at index 0 with keys:
      - images: Tensor(V, 3, S, S) in [0,1]
      - poses:  Tensor(V, 4, 4)
    """

    def __init__(self, data_root: str, image_size: int = 256, downscale: int | None = None):
        self.data_root = str(Path(data_root).resolve())
        self.image_size = int(image_size)
        self.downscale = int(downscale) if downscale is not None else None
        self.final_size = self.downscale if self.downscale is not None else self.image_size
        assert self.final_size > 0, "image_size/downscale must be positive"
        
        # Load file lists
        self.image_paths = self.sorted_image_paths(self.data_root)
        self.poses = self.load_poses_json(self.data_root)
        V_img = len(self.image_paths)
        V_pose = int(self.poses.shape[0])
        assert V_img == V_pose, f"Number of images ({V_img}) != number of poses ({V_pose})"
        if V_img < 2:
            warnings.warn("Less than 2 views; results may be poor.")
        
        # Preload and resize images into a tensor (V,3,S,S)
        tfm = T.Compose([
            T.Resize((self.final_size, self.final_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
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
        return {"images": self.images, "poses": self.poses}

    def load_poses_json(self, path: str | Path) -> torch.Tensor:
        """Read poses.json and return Tensor(V, 4, 4), dtype float32.

        Expects a JSON with key "poses" containing a list of 4x4 matrices.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "poses.json"
        assert path.exists(), f"poses.json not found at: {path}"
        
        with path.open("r") as f:
            data = json.load(f)
        assert "poses" in data, "poses.json missing key 'poses'"
        
        poses_list = data["poses"]
        assert isinstance(poses_list, list) and len(poses_list) > 0, "poses must be a non-empty list"
        
        # Validate each pose
        tensors = []
        for i, p in enumerate(poses_list):
            t = torch.tensor(p, dtype=torch.float32)
            if t.shape != (4, 4):
                raise ValueError(f"pose index {i} has shape {tuple(t.shape)}, expected (4,4)")
            tensors.append(t)
        
        poses = torch.stack(tensors, dim=0)
        return poses


    def sorted_image_paths(self, root: str | Path, exts: List[str] | None = None) -> list[str]:
        """Return sorted image paths under root/images.

        Args:
            root: scene root directory
            exts: allowed extensions (case-insensitive)
        Returns:
            List of absolute string paths sorted lexicographically.
        """
        root = Path(root)
        images_dir = root / "images"
        assert images_dir.exists(), f"images/ not found under: {root}"

        if exts is None:
            exts = [".png", ".jpg", ".jpeg", ".webp"]
        exts = [e.lower() for e in exts]

        files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        assert files, f"No images found in {images_dir} with extensions {exts}"

        files_sorted = sorted(files, key=lambda p: p.name)
        return [str(p.resolve()) for p in files_sorted]