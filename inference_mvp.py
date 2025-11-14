"""Inference script (SD-Turbo, paired JSON only).

Usage example:
    python inference_mvp.py --sd-turbo-id stabilityai/sd-turbo \
            --paired-json fern_paired.json --ckpt checkpoints/mvp_sd_turbo_step_0100.pt
"""
import argparse
import os
from pathlib import Path
import torch
import torchvision.transforms as T

from mvp.model_mvp import MVPSDTurbo
from mvp.dataset_mvp import MVPPairedDataset

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference DIFIX3D MVP (SD-Turbo, paired JSON only)")
    p.add_argument("--sd-turbo-id", type=str, default="stabilityai/sd-turbo")
    p.add_argument("--paired-json", type=str, required=True, help="Paired dataset JSON path")
    p.add_argument("--sample-id", type=str, default=None, help="Sample key inside paired JSON (default: first entry)")
    p.add_argument("--ckpt", type=str, required=False, help="Checkpoint path (optional for sd-turbo frozen)")
    p.add_argument("--sigma-latent", type=float, default=0.3, help="(unused) kept for CLI parity")
    p.add_argument("--out", type=str, default="outputs/infer.png")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--prompt", type=str, default="a photo", help="Text prompt")
    p.add_argument("--lora-rank-vae", type=int, default=4, help="LoRA rank for VAE decoder (should match training)")
    p.add_argument("--enable-xformers", action="store_true", help="Use xFormers memory-efficient attention if available")
    return p


def ensure_dir(path: str | Path):
    os.makedirs(Path(path).parent, exist_ok=True)

def save_image(t: torch.Tensor, path: str | Path):
    t = t.detach().cpu().clamp(0, 1)
    if t.dim() == 4:
        t = t[0]
    img = T.ToPILImage()(t)
    img.save(path)

def run_sd_turbo(args):
    ds = MVPPairedDataset(args.paired_json, split='test', image_size=0)
    key = ds.keys[0] if args.sample_id is None else args.sample_id
    idx = ds.keys.index(key)
    batch = ds[idx]
    images = batch['conditioning_pixel_values'].unsqueeze(0).to(args.device)
    poses = torch.eye(4, device=args.device).view(1,1,4,4).repeat(images.shape[0], images.shape[1],1,1)
    model = MVPSDTurbo(
        sd_turbo_id=args.sd_turbo_id,
        device=args.device,
        lora_rank_vae=args.lora_rank_vae,
        enable_xformers=args.enable_xformers,
    ).to(args.device)
    if args.enable_xformers:
        try:
            backend = model.get_attention_backend()
            print(f"[SD-Turbo] Attention backend (inference): {backend}")
        except Exception:
            pass
    if args.ckpt and os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location=args.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[SD-Turbo] Loaded ckpt", args.ckpt, "missing", missing, "unexpected", unexpected)
    model.set_eval()
    with torch.no_grad():
        out = model(images, poses, prompt=args.prompt)  # (B,V,3,H,W) in [-1,1]
    ensure_dir(args.out)
    # Paired mode: primary slice 0 is the corrupted view we reconstruct
    target = out[:, 0]
    target = ((target + 1) / 2).clamp(0, 1)
    save_image(target, args.out)
    print("Saved sd-turbo inference to", args.out)

def main():
    args = build_argparser().parse_args()
    run_sd_turbo(args)

if __name__ == "__main__":
    main()