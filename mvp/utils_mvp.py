"""Utilities for DIFIX3D MVP (Step 15.2).

Contains: pose loading and image path listing.
"""
from __future__ import annotations

from typing import List
import json
from pathlib import Path

import torch


def load_poses_json(path: str | Path) -> torch.Tensor:
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


def sorted_image_paths(root: str | Path, exts: List[str] | None = None) -> list[str]:
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