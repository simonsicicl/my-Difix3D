#!/usr/bin/env python3
"""
Convert LLFF scenes (nerf_llff_data/<scene>/) into DIFIX3D MVP format:

Input (per scene):
  nerf_llff_data/<scene>/
    images/ or images_*/
    poses_bounds.npy  # shape (N, 17), first 15 -> (3,5): [R|t|hwf]

Output (per scene):
  data/<scene>/
    images/
      000.png, 001.png, ... (sorted by source filename)
    poses.json         # {"poses": [4x4 camera-to-world matrices (row-major)]}

Notes:
  - We interpret the first 3x4 of poses_bounds as camera-to-world (c2w).
  - We ignore intrinsics (H, W, focal) for now; dataset/adapter don't require them.
  - Images are re-encoded to PNG for consistency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import List

import numpy as np
from PIL import Image


IMG_EXTS = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]


def find_images_dir(scene_dir: Path) -> Path:
    # Prefer images/, else pick the highest-resolution images_* folder if available
    cand = scene_dir / "images"
    if cand.exists() and cand.is_dir():
        return cand
    # find images_* dirs
    subs = [p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("images_")]
    if not subs:
        raise FileNotFoundError(f"No images/ or images_*/ under {scene_dir}")
    # Heuristic: pick the one with the largest suffix number
    def rank(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return 0
    subs.sort(key=rank, reverse=True)
    return subs[0]


def list_images(images_dir: Path) -> List[Path]:
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix in IMG_EXTS]
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    files = sorted(files, key=lambda p: p.name)
    return files


def load_llff_c2w(poses_bounds_path: Path, n_imgs: int) -> np.ndarray:
    arr = np.load(str(poses_bounds_path))  # (N,17)
    if arr.ndim == 3:  # sometimes saved as (3,5,N)
        arr = arr.reshape(-1, 17)
    assert arr.shape[1] == 17, f"Unexpected poses_bounds shape {arr.shape}"
    poses = arr[:, :15].reshape(-1, 3, 5)  # (N,3,5)
    c2w_3x4 = poses[:, :, :4]  # (N,3,4)
    N = c2w_3x4.shape[0]
    if N != n_imgs:
        # Align lengths by min; warn user
        m = min(N, n_imgs)
        print(f"[warn] poses ({N}) and images ({n_imgs}) count differ; truncating to {m}")
        c2w_3x4 = c2w_3x4[:m]
    mats = []
    for i in range(c2w_3x4.shape[0]):
        M = np.eye(4, dtype=np.float32)
        M[:3, :4] = c2w_3x4[i]
        mats.append(M)
    return np.stack(mats, axis=0)  # (M,4,4)


def convert_scene(scene_dir: Path, out_root: Path, force: bool = False, reencode: bool = True) -> None:
    images_dir = find_images_dir(scene_dir)
    image_files = list_images(images_dir)
    pb_path = scene_dir / "poses_bounds.npy"
    if not pb_path.exists():
        raise FileNotFoundError(f"poses_bounds.npy not found under {scene_dir}")

    poses = load_llff_c2w(pb_path, len(image_files))  # (M,4,4)

    scene_name = scene_dir.name
    out_scene = out_root / scene_name
    out_images = out_scene / "images"
    out_scene.mkdir(parents=True, exist_ok=True)
    if out_images.exists() and force:
        shutil.rmtree(out_images)
    out_images.mkdir(parents=True, exist_ok=True)

    # Write images as 000.png, 001.png, ... following sorted order
    count = poses.shape[0]
    for idx in range(count):
        src = image_files[idx]
        dst = out_images / f"{idx:03d}.png"
        if reencode:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.save(dst, format="PNG")
        else:
            shutil.copy2(src, dst)

    # Save poses.json
    poses_list = poses.tolist()
    with (out_scene / "poses.json").open("w") as f:
        json.dump({"poses": poses_list}, f)

    print(f"Converted {scene_name}: {count} images -> {out_scene}")


def main():
    ap = argparse.ArgumentParser(description="LLFF -> DIFIX3D MVP scene converter")
    ap.add_argument("--llff-root", type=str, required=True, help="Path to nerf_llff_data root")
    ap.add_argument("--out-root", type=str, default="data", help="Output root directory")
    ap.add_argument("--scenes", type=str, nargs="*", default=None, help="Specific scene names to convert; default: all under llff-root")
    ap.add_argument("--no-reencode", action="store_true", help="Copy files instead of re-encoding to PNG")
    ap.add_argument("--force", action="store_true", help="Overwrite existing scene/images directory")
    args = ap.parse_args()

    llff_root = Path(args.llff_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.scenes:
        scene_dirs = [llff_root / s for s in args.scenes]
    else:
        scene_dirs = [p for p in llff_root.iterdir() if p.is_dir()]
    scene_dirs = sorted(scene_dirs, key=lambda p: p.name)

    for sd in scene_dirs:
        convert_scene(sd, out_root, force=args.force, reencode=(not args.no_reencode))


if __name__ == "__main__":
    main()
