# rs_baseline_inference.py
"""Baseline inference: generate post-images from pure noise with multi-condition ControlNet.

Conditions on (pre_image, pre_segmap, post_segmap) fused into a single ControlNet
input. Uses standard SD3 denoising (not FlowEdit ODE).

Output format matches rs_inference.py for direct comparison with rs_evaluate.py.

Usage:
    python rs_baseline_inference.py \
        --hiucd_root /path/to/Hi-UCD \
        --controlnet_path ./checkpoints/sd3_baseline_controlnet/checkpoint-1000 \
        --output_dir ./outputs/rs_baseline
"""

import argparse
import csv
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import SD3ControlNetModel, StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from rs_data.hiucd import (
    hiucd_segmap_to_rgb,
    hiucd_segmap_to_text,
    parse_hiucd_mask,
)


class ConditionFuser(nn.Module):
    """Fuses three VAE-encoded conditions (48-ch) into ControlNet input (16-ch)."""

    def __init__(self, in_channels=48, out_channels=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, pre_img_latent, pre_seg_latent, post_seg_latent):
        cat = torch.cat([pre_img_latent, pre_seg_latent, post_seg_latent], dim=1)
        return self.proj(cat)


def inference_autocast(device, dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def encode_with_vae(vae, image_tensor):
    """Encode a normalized image tensor into the SD3 VAE latent space."""
    device = image_tensor.device
    dtype = image_tensor.dtype
    with inference_autocast(device, dtype), torch.inference_mode():
        latent = vae.encode(image_tensor).latent_dist.mode()
    return (latent - vae.config.shift_factor) * vae.config.scaling_factor


def load_and_encode_image(vae, image_processor, image_path, device, dtype):
    """Load an RGB image and encode to VAE latent."""
    image = Image.open(image_path).convert("RGB")
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_tensor = image_processor.preprocess(image).to(device=device, dtype=dtype)
    return encode_with_vae(vae, image_tensor)


def load_and_encode_segmap(vae, segmap_rgb, resolution, device, dtype):
    """Convert segmap RGB array to VAE latent."""
    seg_img = Image.fromarray(segmap_rgb).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    seg_tensor = transform(seg_img).unsqueeze(0).to(device=device, dtype=dtype)
    return encode_with_vae(vae, seg_tensor)


@torch.no_grad()
def generate_from_noise(
    pipe, scheduler, controlnet, condition_fuser,
    controlnet_cond,
    prompt, negative_prompt,
    latent_shape,
    num_inference_steps, guidance_scale, controlnet_conditioning_scale,
    device, dtype,
):
    """Standard SD3 denoising from pure noise with fused ControlNet conditioning."""

    timesteps, num_steps = retrieve_timesteps(scheduler, num_inference_steps, device)

    # Encode text
    pipe._guidance_scale = guidance_scale
    (
        prompt_embeds, negative_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    if pipe.do_classifier_free_guidance:
        prompt_embeds_cat = torch.cat([negative_prompt_embeds, prompt_embeds])
        pooled_cat = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
    else:
        prompt_embeds_cat = prompt_embeds
        pooled_cat = pooled_prompt_embeds

    # Start from pure noise
    latents = torch.randn(latent_shape, device=device, dtype=dtype)

    # Denoising loop
    for t in tqdm(timesteps, desc="Denoising", leave=False):
        latent_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
        timestep = t.expand(latent_input.shape[0])

        cond_input = torch.cat([controlnet_cond] * 2) if pipe.do_classifier_free_guidance else controlnet_cond

        # ControlNet
        controlnet_block_samples = controlnet(
            hidden_states=latent_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds_cat,
            pooled_projections=pooled_cat,
            controlnet_cond=cond_input,
            conditioning_scale=controlnet_conditioning_scale,
            return_dict=False,
        )[0]

        # Transformer with ControlNet residuals
        noise_pred = pipe.transformer(
            hidden_states=latent_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds_cat,
            pooled_projections=pooled_cat,
            block_controlnet_hidden_states=controlnet_block_samples,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # CFG
        if pipe.do_classifier_free_guidance:
            pred_uncond, pred_cond = noise_pred.chunk(2)
            noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        # Euler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def main():
    parser = argparse.ArgumentParser(description="Baseline ControlNet inference for RS change generation")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--hiucd_root", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to checkpoint dir (contains ControlNet + condition_fuser.pt)")
    parser.add_argument("--output_dir", type=str, default="./outputs/rs_baseline")
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load models
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype,
    ).to(device)
    scheduler = pipe.scheduler

    controlnet = SD3ControlNetModel.from_pretrained(
        args.controlnet_path, torch_dtype=weight_dtype,
    ).to(device)
    controlnet.eval()

    condition_fuser = ConditionFuser(in_channels=48, out_channels=16)
    fuser_path = os.path.join(args.controlnet_path, "condition_fuser.pt")
    condition_fuser.load_state_dict(torch.load(fuser_path, map_location="cpu"))
    condition_fuser.to(device=device, dtype=weight_dtype)
    condition_fuser.eval()

    # Iterate Hi-UCD pairs
    split_dir = Path(args.hiucd_root) / args.split
    pre_img_dir = split_dir / "image" / "2018"
    post_img_dir = split_dir / "image" / "2019"
    mask_dir = split_dir / "mask" / "2018_2019"

    mask_files = sorted(mask_dir.glob("*.png"))
    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    for mask_path in tqdm(mask_files, desc="Generating"):
        stem = mask_path.stem
        pre_img_path = pre_img_dir / f"{stem}.png"
        post_img_path = post_img_dir / f"{stem}.png"

        if not pre_img_path.exists() or not post_img_path.exists():
            print(f"Skipping {stem}: missing image files")
            continue

        # Encode pre-image
        pre_img_lat = load_and_encode_image(
            pipe.vae, pipe.image_processor, str(pre_img_path), device, weight_dtype,
        )

        # Parse mask → segmaps
        mask_rgb = np.array(Image.open(mask_path))
        pre_seg_np, post_seg_np, _ = parse_hiucd_mask(mask_rgb)
        pre_seg_rgb = hiucd_segmap_to_rgb(pre_seg_np)
        post_seg_rgb = hiucd_segmap_to_rgb(post_seg_np)

        vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
        resolution = min(pre_img_lat.shape[2] * vae_scale_factor, pre_img_lat.shape[3] * vae_scale_factor)

        pre_seg_lat = load_and_encode_segmap(pipe.vae, pre_seg_rgb, resolution, device, weight_dtype)
        post_seg_lat = load_and_encode_segmap(pipe.vae, post_seg_rgb, resolution, device, weight_dtype)

        # Fuse conditions
        controlnet_cond = condition_fuser(pre_img_lat, pre_seg_lat, post_seg_lat)

        # Text prompt from post segmap
        text_post = hiucd_segmap_to_text(post_seg_np)

        # Generate from noise
        x0_tar = generate_from_noise(
            pipe, scheduler, controlnet, condition_fuser,
            controlnet_cond,
            prompt=text_post,
            negative_prompt=args.negative_prompt,
            latent_shape=pre_img_lat.shape,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            device=device, dtype=weight_dtype,
        )

        # Decode
        x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with inference_autocast(device, weight_dtype), torch.inference_mode():
            image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
        image_tar = pipe.image_processor.postprocess(image_tar)

        # Save
        out_path = os.path.join(args.output_dir, f"{stem}_generated_post.png")
        image_tar[0].save(out_path)

        results.append({
            "stem": stem,
            "pre_img": str(pre_img_path),
            "post_img_real": str(post_img_path),
            "post_img_generated": out_path,
            "flowedit_src_prompt": "",
            "flowedit_tar_prompt": text_post,
            "prompt_mode": "baseline_controlnet",
        })

    # Save results manifest (compatible with rs_evaluate.py)
    manifest_path = os.path.join(args.output_dir, "results.csv")
    if results:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"Done. {len(results)} images generated. Results: {manifest_path}")


if __name__ == "__main__":
    main()
