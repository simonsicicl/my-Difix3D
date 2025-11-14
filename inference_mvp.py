"""Inference script (SD-Turbo only, single-step latent denoising).

Usage example:
    python inference_mvp.py --sd-turbo-id stabilityai/sd-turbo \
            --data-root example_scene --ckpt checkpoints/mvp_sd_turbo_step_0100.pt --target-index 0 --sigma-latent 0.3
"""
import argparse
import os
from pathlib import Path
import torch
import torchvision.transforms as T

from mvp.model_mvp import MVPSDTurbo
from mvp.dataset_mvp import MVPSceneDataset

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference DIFIX3D MVP (SD-Turbo)")
    p.add_argument("--sd-turbo-id", type=str, default="stabilityai/sd-turbo")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=False, help="Checkpoint path (optional for sd-turbo frozen)")
    p.add_argument("--target-index", type=int, default=0, help="Target view index to reconstruct")
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
    ds = MVPSceneDataset(args.data_root)
    item = ds[0]
    images = item["images"].unsqueeze(0).to(args.device)
    poses = item["poses"].unsqueeze(0).to(args.device)  # adapt shape (B,V,4,4)
    # Validate target index against available views
    V = images.shape[1]
    if not (0 <= args.target_index < V):
        raise IndexError(f"target_index {args.target_index} is out of bounds for available views (0..{V-1}).\n"
                         f"Hint: example_scene currently has {V} views. Pick a target index in [0,{V-1}].")
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
    # Select target view and save in [0,1]
    target = out[:, args.target_index]
    target = ((target + 1) / 2).clamp(0, 1)
    save_image(target, args.out)
    print("Saved sd-turbo inference to", args.out)

def main():
    args = build_argparser().parse_args()
    run_sd_turbo(args)

if __name__ == "__main__":
    main()