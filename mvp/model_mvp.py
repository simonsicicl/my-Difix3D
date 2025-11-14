"""
Model implementation for DIFIX3D MVP (SD-Turbo only)

This module exposes MVPSDTurbo: SD‑Turbo latent single-step denoising with
prompt conditioning via cross-attention. Legacy MVPModel/baseline UNet has
been removed per the project direction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from einops import rearrange, repeat
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel
from typing import Optional

class MVPSDTurbo(nn.Module):
    """SD-Turbo based single-step latent reconstruction with multi-view conditioning.

    Forward:
        images: (B,V,3,H,W)
        poses:  (B,V,4,4)
        target_index: int (which view to reconstruct)
        sigma_latent: float noise scale in latent space

    Returns dict:
        {
        'eps_hat': predicted noise in latent space,
        'eps': ground-truth noise tensor,
        'x0_hat': reconstructed RGB in [0,1],
        'x0': target RGB ground truth,
        }
    """

    def __init__(
        self,
        sd_turbo_id: str = "stabilityai/sd-turbo",
        lora_rank_vae: int = 4,
        device: str | torch.device = "cuda",
        timestep: int = 999,
        enable_xformers: bool | None = None,
    ):
        super().__init__()
        self.sd_turbo_id = sd_turbo_id
        self.device_ref = device
        self.vae = AutoencoderKL.from_pretrained(sd_turbo_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(sd_turbo_id, subfolder="unet")
        self.scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)
        self._setup_vae_skip_and_lora(lora_rank_vae=lora_rank_vae)

        # Initialize text conditioning modules (temporarily ignore multi-view adapter path)
        # Text-only path for now: do not instantiate adapter to avoid unnecessary state dict keys
        self.tokenizer = AutoTokenizer.from_pretrained(sd_turbo_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_turbo_id, subfolder="text_encoder")
        self.text_encoder.requires_grad_(False)

        # One-step DDPM scheduler (init once)
        self.sched = DDPMScheduler.from_pretrained(sd_turbo_id, subfolder="scheduler")
        self.sched.set_timesteps(1, device=device)
        # move internal buffers to device (mimic src/model.py behavior)
        self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(device)
        self.timesteps = torch.tensor([timestep], device=device, dtype=torch.long)

        # Optionally enable memory-efficient attention via xFormers when requested
        if enable_xformers:
            try:
                self.enable_xformers()
            except Exception:
                pass

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        if hasattr(self.vae, 'decoder'):
            for name in ['skip_conv_1','skip_conv_2','skip_conv_3','skip_conv_4']:
                if hasattr(self.vae.decoder, name):
                    getattr(self.vae.decoder, name).requires_grad_(True)

    def enable_xformers(self) -> bool:
        """Enable xFormers memory-efficient attention on UNet if available.

        Returns True on success, False otherwise.
        """
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            return True
        except Exception:
            return False

    def get_attention_backend(self) -> str:
        """Best-effort inspection of current attention backend used by UNet.

        Returns one of {"xformers", "sdpa", "default"}.
        """
        # diffusers>=0.14 exposes attn_processors mapping
        try:
            from diffusers.models.attention_processor import (
                XFormersAttnProcessor,
                AttnProcessor2_0,
                AttnProcessor,
            )
            procs = getattr(self.unet, "attn_processors", None)
            if isinstance(procs, dict) and procs:
                any_xf = any(isinstance(p, XFormersAttnProcessor) for p in procs.values())
                if any_xf:
                    return "xformers"
                any_sdp = any(isinstance(p, AttnProcessor2_0) for p in procs.values())
                if any_sdp:
                    return "sdpa"
                any_def = any(isinstance(p, AttnProcessor) for p in procs.values())
                if any_def:
                    return "default"
        except Exception:
            pass
        # Fallback heuristic
        try:
            # If xformers is importable and was requested earlier, assume active
            import xformers  # noqa: F401
            return "xformers?"
        except Exception:
            return "unknown"

    def _setup_vae_skip_and_lora(self, lora_rank_vae: int = 4):
        """Add lightweight trainable components to VAE decoder:
        - Monkey-patch encoder/decoder forward to expose skip activations.
        - Add 4 skip 1x1 convs on decoder; only these and LoRA weights are trainable.
        - Attach LoRA to selected decoder modules.
        """
        vae = self.vae

        # 1) Monkey-patch encoder forward to cache down-block activations
        def enc_fwd(self_enc, sample):
            sample = self_enc.conv_in(sample)
            l_blocks = []
            for down_block in self_enc.down_blocks:
                l_blocks.append(sample)
                sample = down_block(sample)
            sample = self_enc.mid_block(sample)
            sample = self_enc.conv_norm_out(sample)
            sample = self_enc.conv_act(sample)
            sample = self_enc.conv_out(sample)
            self_enc.current_down_blocks = l_blocks
            return sample
        vae.encoder.forward = enc_fwd.__get__(vae.encoder, vae.encoder.__class__)

        # 2) Add skip convs and monkey-patch decoder forward to inject skips
        # Create skip 1x1 convs matching SD1.5 channel sizes
        vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        vae.decoder.ignore_skip = False
        vae.decoder.gamma = 1.0

        def dec_fwd(self_dec, sample, latent_embeds=None):
            sample = self_dec.conv_in(sample)
            # middle
            sample = self_dec.mid_block(sample, latent_embeds)
            # add skip connections if available
            if not getattr(self_dec, 'ignore_skip', False):
                skip_convs = [self_dec.skip_conv_1, self_dec.skip_conv_2, self_dec.skip_conv_3, self_dec.skip_conv_4]
                incoming = getattr(self_dec, 'incoming_skip_acts', None)
                if incoming is not None:
                    for idx, up_block in enumerate(self_dec.up_blocks):
                        skip_in = skip_convs[idx](incoming[::-1][idx] * self_dec.gamma)
                        # If spatial shapes differ (e.g., VAE tiling or odd sizes), resize skip to match
                        if skip_in.shape[-2:] != sample.shape[-2:]:
                            skip_in = F.interpolate(skip_in, size=sample.shape[-2:], mode='bilinear', align_corners=False)
                        sample = sample + skip_in
                        sample = up_block(sample, latent_embeds)
                else:
                    for up_block in self_dec.up_blocks:
                        sample = up_block(sample, latent_embeds)
            else:
                for up_block in self_dec.up_blocks:
                    sample = up_block(sample, latent_embeds)
            # post-process
            if latent_embeds is None:
                sample = self_dec.conv_norm_out(sample)
            else:
                sample = self_dec.conv_norm_out(sample, latent_embeds)
            sample = self_dec.conv_act(sample)
            sample = self_dec.conv_out(sample)
            return sample

        vae.decoder.forward = dec_fwd.__get__(vae.decoder, vae.decoder.__class__)

        # 3) Freeze policy: freeze VAE encoder fully; train only LoRA weights and skip convs in decoder
        vae.encoder.requires_grad_(False)
        vae.decoder.requires_grad_(False)
        # ensure skip convs are trainable
        vae.decoder.skip_conv_1.requires_grad_(True)
        vae.decoder.skip_conv_2.requires_grad_(True)
        vae.decoder.skip_conv_3.requires_grad_(True)
        vae.decoder.skip_conv_4.requires_grad_(True)

        # 4) Attach LoRA adapters to VAE decoder selected modules
        target_suffixes = [
            "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "to_k", "to_q", "to_v", "to_out.0",
        ]
        target_modules = []
        for name, module in vae.named_modules():
            if 'decoder' in name and any(name.endswith(sfx) for sfx in target_suffixes):
                target_modules.append(name)
        if len(target_modules) > 0:
            lcfg = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules)
            vae.add_adapter(lcfg, adapter_name="vae_skip")
            # Only LoRA weights in decoder should be trainable (plus skip convs set above)
            for n, p in vae.named_parameters():
                if 'lora' in n:
                    p.requires_grad = True
                elif 'decoder.skip_conv_' in n:
                    p.requires_grad = True
                else:
                    # keep the rest of VAE frozen
                    pass

    @torch.no_grad()
    def encode_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Encode RGB to latents. Accepts (B,3,H,W) or (B,V,3,H,W).
        Uses einops.rearrange to flatten/unflatten when views are present.
        """
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            x = rearrange(x, 'b v c h w -> (b v) c h w')
            x = (x * 2 - 1)
            z = self.vae.encode(x).latent_dist.sample() * self.scaling_factor
            z = rearrange(z, '(b v) c h w -> b v c h w', b=B, v=V)
            return z
        else:
            x = (x * 2 - 1)
            latents = self.vae.encode(x).latent_dist.sample() * self.scaling_factor
            return latents

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        # If skip is enabled, pass encoder activations to decoder before decode()
        if hasattr(self.vae.encoder, 'current_down_blocks'):
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        # Disable skip injection when VAE tiling is active to avoid shape mismatches
        if getattr(self.vae, 'use_tiling', False):
            self.vae.decoder.ignore_skip = True
        else:
            self.vae.decoder.ignore_skip = False
        if z.dim() == 5:
            B, V, C, H, W = z.shape
            zf = rearrange(z, 'b v c h w -> (b v) c h w')
            x = self.vae.decode(zf / self.scaling_factor).sample
            x = rearrange(x, '(b v) c h w -> b v c h w', b=B, v=V)
        else:
            x = self.vae.decode(z / self.scaling_factor).sample
        x = (x + 1) / 2
        return x.clamp(0, 1)

    def forward(
        self,
        images: torch.Tensor,
        poses: torch.Tensor,
        target_index: int = 0,
        sigma_latent: float = 0.3,
        predict_eps: bool = True,
        prompt: str | None = None,
        prompt_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Text-conditioned single-step denoising like src/model.py Difix.forward.

        Flow:
          - Encode text -> caption_enc (B,N,C)
          - Flatten views: (B,V,3,H,W) -> ((B*V),3,H,W)
          - VAE encode to latents z (no extra noise add), scale by scaling_factor
          - Repeat caption_enc over views: -> ((B*V),N,C)
          - UNet predict at self.timesteps; scheduler single step to get denoised latents
          - Inject VAE skip activations if available and decode; clamp to [-1,1]
          - Unflatten views back to (B,V,3,H,W)
        Returns output_image in [-1,1].
        """
        device = images.device
        B, V, C, H, W = images.shape
        assert C == 3

        # Encode text prompt/tokens; ensure batch size B alignment
        if (prompt is None) and (prompt_tokens is None):
            prompt = "a photo"
        if prompt is not None:
            # Accept str or list; if str, repeat across batch
            if isinstance(prompt, str):
                prompts = [prompt] * B
            else:
                prompts = list(prompt)
                if len(prompts) != B:
                    # pad or truncate to match B
                    if len(prompts) < B:
                        prompts = (prompts + [prompts[-1]] * (B - len(prompts)))[:B]
                    else:
                        prompts = prompts[:B]
            toks = self.tokenizer(
                prompts,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = toks.input_ids.to(device)
        else:
            input_ids = prompt_tokens.to(device)
            # If provided tokens don't match batch, tile to B
            if input_ids.shape[0] != B:
                input_ids = repeat(input_ids, 'b n -> (b r) n', r=(B // input_ids.shape[0]))
                # In case B isn't multiple, slice
                input_ids = input_ids[:B]
        with torch.no_grad():
            caption_enc = self.text_encoder(input_ids)[0]  # (B,N,C)

        # Flatten views and normalize to [-1,1] before VAE encode (robust to datasets in [0,1])
        x = rearrange(images, 'b v c h w -> (b v) c h w')
        x = x * 2 - 1
        z = self.vae.encode(x).latent_dist.sample() * self.scaling_factor  # ((B*V),4,h,w)

        # Repeat caption encodings across views to match flattened (B*V)
        caption_enc = repeat(caption_enc, 'b n c -> (b v) n c', v=V)

        # UNet predict and one-step denoise
        model_pred = self.unet(z, self.timesteps, encoder_hidden_states=caption_enc).sample
        z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample

        # Skip-connection assisted decode if activations are cached
        if hasattr(self.vae.encoder, 'current_down_blocks'):
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        # Disable skip injection when VAE tiling is active to avoid shape mismatches
        if getattr(self.vae, 'use_tiling', False):
            self.vae.decoder.ignore_skip = True
        else:
            self.vae.decoder.ignore_skip = False
        output_image = (self.vae.decode(z_denoised / self.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=V)
        return output_image

__all__ = [
  'MVPSDTurbo',
]