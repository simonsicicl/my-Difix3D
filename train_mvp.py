"""Training script for DIFIX3D MVP (SD-Turbo only, single-step latent denoising).

This script trains the SD-Turbo path with a multi-view adapter (no baseline UNet).
"""
import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from mvp.model_mvp import MVPSDTurbo
from mvp.dataset_mvp import MVPSceneDataset
from tools.defects import corrupt_batch

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DIFIX3D MVP (SD-Turbo)")
    p.add_argument("--sd-turbo-id", type=str, default="stabilityai/sd-turbo")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--image-size", type=int, default=256, help="Resize images to SxS for training")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--sigma-latent", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--outdir", type=str, default="checkpoints")
    p.add_argument("--random-target", action="store_true", help="Randomize target view each step")
    p.add_argument("--prompt", type=str, default="a photo", help="Text prompt if --cond text")
    p.add_argument("--lora-rank-vae", type=int, default=4, help="LoRA rank for VAE decoder (always enabled)")
    # Memory/perf options
    p.add_argument("--freeze-unet", action="store_true", help="Freeze UNet to save memory (recommended initially)")
    p.add_argument("--enable-ckpt", action="store_true", help="Enable gradient checkpointing for UNet")
    p.add_argument("--enable-vae-tiling", action="store_true", help="Enable VAE tiling+slicing to reduce memory")
    # Defect synthesis options
    p.add_argument("--enable-defects", action="store_true", help="Train with corrupted inputs (DIFIX-style)")
    p.add_argument("--defects", type=str, default="mask,noise,blur,jpeg,occlusion,color",
                   help="Comma-separated defect kinds to apply to inputs")
    p.add_argument("--defect-prob", type=float, default=0.7, help="Per-op probability for applying a defect")
    p.add_argument("--defect-strength", type=float, default=1.0, help="Overall strength scaling for defects [0-1]")
    return p

def ensure_dir(path: str | Path):
    os.makedirs(path, exist_ok=True)

def save_image(t: torch.Tensor, path: str):
    t = t.detach().cpu().clamp(0, 1)
    vutils.save_image(t, path)

def train_sd_turbo(args):
    ds = MVPSceneDataset(args.data_root, image_size=args.image_size)
    item = ds[0]
    images = item["images"].unsqueeze(0).to(args.device)  # (1,V,3,H,W)
    poses = item["poses"].unsqueeze(0).to(args.device)    # (1,V,4,4)
    # Only adapter (and optionally UNet if unfrozen) will train
    model = MVPSDTurbo(
        sd_turbo_id=args.sd_turbo_id,
        device=args.device,
        lora_rank_vae=args.lora_rank_vae,
    ).to(args.device)
    model.set_train()
    # Memory/optimization toggles
    if args.freeze_unet:
        model.unet.eval()
        for p in model.unet.parameters():
            p.requires_grad = False
    if args.enable_ckpt:
        try:
            model.unet.enable_gradient_checkpointing()
            print("[SD-Turbo] Enabled UNet gradient checkpointing")
        except Exception:
            pass
    if args.enable_vae_tiling:
        try:
            model.vae.enable_slicing()
            model.vae.enable_tiling()
            print("[SD-Turbo] Enabled VAE slicing+tiling")
        except Exception:
            pass

    # Gather trainable params AFTER freezing policy applied
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[SD-Turbo] Trainable params: {sum(p.numel() for p in params)/1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    V = images.shape[1]
    defect_kinds = [s.strip() for s in args.defects.split(',') if s.strip()]
    for step in range(1, args.steps + 1):
        target_index = torch.randint(0, V, (1,)).item() if args.random_target else 0
        # Prepare clean GT and optionally corrupted inputs
        gt_imgs = images.clone()
        in_imgs = images
        if args.enable_defects:
            in_imgs = corrupt_batch(in_imgs, kinds=defect_kinds, prob=args.defect_prob, strength=args.defect_strength)
        # Forward now returns reconstructed images in [-1,1] with shape (B,V,3,H,W)
        out_imgs = model(in_imgs, poses, prompt=args.prompt)
        # Prepare GT in the same range [-1,1]
        gt_imgs = gt_imgs * 2 - 1
        # Train on the selected target view to mirror prior behavior
        pred = out_imgs[:, target_index]
        gt = gt_imgs[:, target_index]
        loss = F.mse_loss(pred, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"[SD-Turbo] step={step} loss={loss.item():.4f} target_index={target_index}")
        if step % args.save_every == 0:
            ensure_dir(args.outdir)
            ckpt = os.path.join(args.outdir, f"mvp_sd_turbo_step_{step:04d}.pt")
            torch.save(model.state_dict(), ckpt)
            ensure_dir("outputs")
            # Convert prediction for target view to [0,1] for visualization
            vis = ((pred + 1) / 2).clamp(0, 1)
            save_image(vis, f"outputs/recon_latent_step_{step:04d}.png")
    print("SD-Turbo training done.")

def main():
    args = build_argparser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    train_sd_turbo(args)

if __name__ == "__main__":
    main()
