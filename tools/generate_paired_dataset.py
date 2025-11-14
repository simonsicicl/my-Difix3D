"""Generate paired dataset JSON (mimicking src/PairedDataset) from a multi-view scene.

For each view in a scene we create a record:
  - image: path to corrupted (artifact) version of that view
  - target_image: path to clean original view (same view) OR optionally a fixed clean reference
  - ref_image: optional path to a single chosen clean reference view (shared across all records)
  - prompt: text prompt (default: "a photo")

We also write corrupted images to an artifacts folder so training/inference
can load without regenerating.

Example:
  python tools/generate_paired_dataset.py \
      --scene-root data/fern --out-json paired_fern.json \
      --ref-index 0 --modes sparse,cycle,underfit,cross --artifact-prob 0.7

Splits:
  By default we place all records in "train"; you can specify --val-ratio and --test-ratio
to carve out validation/test splits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import torch
import os

def sorted_image_paths(root: str) -> List[Path]:
    exts = {"png","jpg","jpeg"}
    paths = []
    for p in sorted((Path(root)/"images").glob("*")):
        if p.suffix.lower().lstrip('.') in exts:
            paths.append(p)
    return paths

def _jpeg_cycle(pil_img: Image.Image, cycles: int, q_low: int, q_high: int) -> Image.Image:
    import io
    img = pil_img
    for _ in range(max(1, cycles)):
        q = random.randint(q_low, q_high)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img

def _sparse_reconstruction(t: torch.Tensor, keep_prob: float = 0.3, cell: int = 32) -> torch.Tensor:
    c,h,w = t.shape
    out = t.clone()
    for y in range(0,h,cell):
        for x in range(0,w,cell):
            if random.random() > keep_prob:
                y2=min(y+cell,h); x2=min(x+cell,w)
                patch=out[:,y:y2,x:x2]
                out[:,y:y2,x:x2]=patch.mean(dim=(1,2),keepdim=True)
    return out

def _model_underfitting(pil_img: Image.Image, blur_radius: int = 4, quant_levels: int = 32) -> Image.Image:
    try:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    except Exception:
        pass
    t = TF.to_tensor(pil_img)
    t = torch.round(t * (quant_levels-1)) / (quant_levels-1)
    # simple sharpening kernel
    kernel = torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=torch.float32).view(1,1,3,3)
    t = torch.nn.functional.conv2d(t.unsqueeze(0), kernel.repeat(3,1,1,1), padding=1, groups=3).squeeze(0).clamp(0,1)
    return TF.to_pil_image(t)

def _cross_reference(base: torch.Tensor, others: torch.Tensor, patches: int = 8) -> torch.Tensor:
    c,h,w=base.shape
    out=base.clone()
    if others.numel()==0:
        return out
    for _ in range(patches):
        src=others[random.randrange(0,others.shape[0])]
        ph=random.randint(h//16,h//6)
        pw=random.randint(w//16,w//6)
        y=random.randint(0,h-ph); x=random.randint(0,w-pw)
        sy=random.randint(0,src.shape[1]-ph); sx=random.randint(0,src.shape[2]-pw)
        patch=src[:,sy:sy+ph,sx:sx+pw]
        alpha=0.5
        out[:,y:y+ph,x:x+pw]=out[:,y:y+ph,x:x+pw]*(1-alpha)+patch*alpha
    return out

def corrupt_image(pil_img: Image.Image, modes: List[str], others_tensor: torch.Tensor, prob: float, cycles: int, cross_patches: int) -> Image.Image:
    t = TF.to_tensor(pil_img)
    cur = t.clone()
    for m in modes:
        if random.random() > prob:
            continue
        if m == 'sparse':
            cur = _sparse_reconstruction(cur)
        elif m == 'cycle':
            pil_cur = TF.to_pil_image(cur)
            pil_cur = _jpeg_cycle(pil_cur, cycles=cycles, q_low=40, q_high=70)
            cur = TF.to_tensor(pil_cur)
        elif m == 'underfit':
            pil_cur = TF.to_pil_image(cur)
            pil_cur = _model_underfitting(pil_cur)
            cur = TF.to_tensor(pil_cur)
        elif m == 'cross':
            cur = _cross_reference(cur, others_tensor, patches=cross_patches)
    return TF.to_pil_image(cur.clamp(0,1))

def build_records(scene_root: str, ref_index: int, modes: List[str], prob: float, cycles: int, prompt: str,
                  artifacts_dir: Path, resize: int | None, cross_samples: int, cross_patches: int) -> List[dict]:
    paths = sorted_image_paths(scene_root)
    assert 0 <= ref_index < len(paths), "ref_index out of range"
    records = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if resize:
                im = im.resize((resize, resize), Image.BILINEAR)
            # Build a small on-demand pool of other views for cross-reference to avoid OOM
            if ('cross' in modes) and (cross_samples > 0):
                cand_idxs = [j for j in range(len(paths)) if j != i]
                random.shuffle(cand_idxs)
                pick = cand_idxs[:cross_samples]
                other_tensors = []
                for j in pick:
                    with Image.open(paths[j]) as oim:
                        oim = oim.convert("RGB")
                        if resize:
                            oim = oim.resize((resize, resize), Image.BILINEAR)
                        other_tensors.append(TF.to_tensor(oim))
                others = torch.stack(other_tensors, dim=0) if other_tensors else torch.empty(0)
            else:
                others = torch.empty(0)
            corrupt_pil = corrupt_image(im, modes, others, prob, cycles, cross_patches)
        # Save corrupt image
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        corrupt_path = artifacts_dir / f"{i:05d}.png"
        corrupt_pil.save(corrupt_path)
        record = {
            "image": str(corrupt_path),              # corrupted input
            "target_image": str(p),                  # clean target (same view)
            "ref_image": str(paths[ref_index]),      # global reference clean view
            "prompt": prompt,
        }
        records.append(record)
    return records

def split_records(records: List[dict], val_ratio: float, test_ratio: float, seed: int | None):
    idxs = list(range(len(records)))
    if seed is not None:
        random.seed(seed)
    random.shuffle(idxs)
    n = len(records)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_ids = idxs[:n_val]
    test_ids = idxs[n_val:n_val+n_test]
    train_ids = idxs[n_val+n_test:]
    def subset(id_list):
        return {f"view_{i:05d}": records[i] for i in id_list}
    return subset(train_ids), subset(val_ids), subset(test_ids)

def main():
    ap = argparse.ArgumentParser(description="Generate paired dataset JSON from scene")
    ap.add_argument("--scene-root", required=True, help="Path to scene root containing images/ and poses.json")
    ap.add_argument("--out-json", required=True, help="Output JSON file path")
    ap.add_argument("--ref-index", type=int, default=0, help="Index of clean reference view")
    ap.add_argument("--modes", type=str, default="sparse,cycle,underfit,cross")
    ap.add_argument("--artifact-prob", type=float, default=0.7)
    ap.add_argument("--artifact-cycles", type=int, default=2)
    ap.add_argument("--prompt", type=str, default="a photo")
    ap.add_argument("--resize", type=int, default=256, help="Resize square; <=0 keeps original")
    ap.add_argument("--cross-samples", type=int, default=6, help="Number of other views loaded per image for cross-reference (limits memory)")
    ap.add_argument("--cross-patches", type=int, default=8, help="Number of patches blended in cross-reference mode")
    ap.add_argument("--val-ratio", type=float, default=0.0)
    ap.add_argument("--test-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifacts-subdir", type=str, default="paired_artifacts")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    resize = args.resize if args.resize > 0 else None
    artifacts_dir = Path(args.scene_root)/args.artifacts_subdir
    records = build_records(
        args.scene_root,
        args.ref_index,
        modes,
        args.artifact_prob,
        args.artifact_cycles,
        args.prompt,
        artifacts_dir,
        resize,
        args.cross_samples,
        args.cross_patches,
    )
    train_dict, val_dict, test_dict = split_records(records, args.val_ratio, args.test_ratio, args.seed)
    payload = {"train": train_dict, "val": val_dict, "test": test_dict}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(records)} records to {args.out_json}\nArtifacts saved under {artifacts_dir}")

if __name__ == "__main__":
    main()
