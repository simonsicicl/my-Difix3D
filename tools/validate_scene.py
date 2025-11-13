#!/usr/bin/env python3
"""
Validate DIFIX3D MVP scene folders:

Usage:
  python tools/validate_scene.py --scene data/<scene>
  python tools/validate_scene.py --all data/

Checks:
  - images/ exists and has files
  - poses.json exists and contains list of 4x4 matrices
  - number of images == number of poses
  - optional: verify image sizes are loadable and report common stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from PIL import Image


def validate_one(scene_root: Path) -> bool:
    ok = True
    images_dir = scene_root / "images"
    poses_path = scene_root / "poses.json"
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"[ERR] {scene_root}: images/ not found")
        return False
    if not poses_path.exists():
        print(f"[ERR] {scene_root}: poses.json not found")
        return False
    imgs = sorted([p for p in images_dir.iterdir() if p.is_file()])
    if not imgs:
        print(f"[ERR] {scene_root}: no images in images/")
        return False
    try:
        data = json.loads(poses_path.read_text())
        poses = data.get("poses", None)
        assert isinstance(poses, list) and len(poses) > 0
        shapes = set((len(p), len(p[0])) for p in poses)
        if shapes != {(4, 4)}:
            print(f"[ERR] {scene_root}: poses not 4x4; shapes={shapes}")
            ok = False
    except Exception as e:
        print(f"[ERR] {scene_root}: failed to parse poses.json: {e}")
        return False

    n_img, n_pose = len(imgs), len(poses)
    if n_img != n_pose:
        print(f"[ERR] {scene_root}: images ({n_img}) != poses ({n_pose})")
        ok = False

    # Try load first image to report size
    try:
        with Image.open(imgs[0]) as im:
            w, h = im.size
        print(f"[OK ] {scene_root.name}: {n_img} views, image size {w}x{h}")
    except Exception as e:
        print(f"[WARN] {scene_root}: failed to open first image: {e}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Validate DIFIX3D MVP scenes")
    ap.add_argument("--scene", type=str, help="Path to one scene root (contains images/ and poses.json)")
    ap.add_argument("--all", type=str, help="Validate all subfolders under this data root", nargs='?')
    args = ap.parse_args()

    any_fail = False
    if args.scene:
        root = Path(args.scene)
        ok = validate_one(root)
        any_fail = any_fail or (not ok)
    elif args.all:
        data_root = Path(args.all)
        for p in sorted([q for q in data_root.iterdir() if q.is_dir()]):
            ok = validate_one(p)
            any_fail = any_fail or (not ok)
    else:
        ap.error("Specify --scene <path> or --all <data_root>")

    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
