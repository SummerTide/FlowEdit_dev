"""Validate SD3 ControlNet checkpoint by generating images from segmentation maps.

Tests pure ControlNet generation quality (from noise, not FlowEdit).
Uses custom inference loop to feed RGB segmaps directly to ControlNet
(bypassing the pipeline's default VAE encoding of control images).

Usage:
    python validate_controlnet.py \
        --controlnet_path ./checkpoints/sd3_controlnet_rs/checkpoint-2000 \
        --hiucd_root /path/to/Hi-UCD \
        --output_dir ./outputs/controlnet_val \
        --num_samples 20 \
        --device_number 0
"""

import argparse
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, SD3ControlNetModel, StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from rs_data.hiucd import parse_hiucd_mask, hiucd_segmap_to_rgb, hiucd_segmap_to_text


def generate_with_controlnet(pipe, controlnet, seg_rgb_tensor, prompt, device, weight_dtype,
                             num_inference_steps=28, guidance_scale=7.0):
    """Generate an image using SD3 + ControlNet with RGB segmap input (no VAE encoding of segmap).

    Args:
        pipe: StableDiffusion3Pipeline (for VAE, text encoders, transformer, scheduler)
        controlnet: SD3ControlNetModel with 3-ch pos_embed_input
        seg_rgb_tensor: (1, 3, H, W) tensor in [0, 1]
        prompt: text prompt string
    """
    # Encode text
    pipe._guidance_scale = guidance_scale
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        negative_prompt="",
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # Prepare timesteps
    scheduler = pipe.scheduler
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)

    # Prepare latent noise
    num_channels_latents = pipe.transformer.config.in_channels
    h, w = seg_rgb_tensor.shape[2], seg_rgb_tensor.shape[3]
    latent_h, latent_w = h // pipe.vae_scale_factor, w // pipe.vae_scale_factor
    latents = torch.randn(1, num_channels_latents, latent_h, latent_w, device=device, dtype=weight_dtype)

    # Prepare controlnet_cond — RGB segmap, NOT through VAE
    controlnet_cond = seg_rgb_tensor.to(device=device, dtype=weight_dtype)

    # Denoising loop
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
        timestep = t.expand(latent_model_input.shape[0])

        # ControlNet — same cond for uncond and cond
        ctrl_cond = torch.cat([controlnet_cond] * 2) if pipe.do_classifier_free_guidance else controlnet_cond

        controlnet_block_samples = controlnet(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]) if pipe.do_classifier_free_guidance else prompt_embeds,
            pooled_projections=torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]) if pipe.do_classifier_free_guidance else pooled_prompt_embeds,
            controlnet_cond=ctrl_cond,
            return_dict=False,
        )[0]

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]) if pipe.do_classifier_free_guidance else prompt_embeds,
            pooled_projections=torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]) if pipe.do_classifier_free_guidance else pooled_prompt_embeds,
            block_controlnet_hidden_states=controlnet_block_samples,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Decode
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image)[0]
    return image


def main():
    parser = argparse.ArgumentParser(description="Validate SD3 ControlNet on Hi-UCD")
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--hiucd_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/controlnet_val")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    torch.manual_seed(args.seed)

    # Load base pipeline
    print("Loading SD3 pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=weight_dtype,
    ).to(device)

    # Load ControlNet (with RGB pos_embed_input)
    print(f"Loading ControlNet from {args.controlnet_path}...")
    from train_controlnet_sd3_rs import load_rgb_controlnet
    controlnet = load_rgb_controlnet(args.controlnet_path, dtype=weight_dtype).to(device)
    controlnet.eval()

    # Segmap transform — resize to match latent resolution, keep as [0, 1] RGB
    seg_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])

    # Collect mask files
    mask_dir = os.path.join(args.hiucd_root, args.split, "mask", "2018_2019")
    pre_img_dir = os.path.join(args.hiucd_root, args.split, "image", "2018")
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    mask_files = mask_files[:args.num_samples]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {len(mask_files)} samples...")
    for mask_name in tqdm(mask_files):
        stem = os.path.splitext(mask_name)[0]

        # Parse mask
        mask_rgb = np.array(Image.open(os.path.join(mask_dir, mask_name)))
        seg_pre, seg_post, _ = parse_hiucd_mask(mask_rgb)

        # Load real image for comparison
        real_pre = Image.open(os.path.join(pre_img_dir, mask_name)).convert("RGB")

        # Generate from pre segmap
        seg_pre_rgb_img = Image.fromarray(hiucd_segmap_to_rgb(seg_pre))
        seg_pre_tensor = seg_transform(seg_pre_rgb_img).unsqueeze(0)
        prompt_pre = hiucd_segmap_to_text(seg_pre)

        with torch.no_grad():
            gen_pre = generate_with_controlnet(
                pipe, controlnet, seg_pre_tensor, prompt_pre, device, weight_dtype,
                num_inference_steps=args.num_inference_steps,
            )

        # Generate from post segmap
        seg_post_rgb_img = Image.fromarray(hiucd_segmap_to_rgb(seg_post))
        seg_post_tensor = seg_transform(seg_post_rgb_img).unsqueeze(0)
        prompt_post = hiucd_segmap_to_text(seg_post)

        with torch.no_grad():
            gen_post = generate_with_controlnet(
                pipe, controlnet, seg_post_tensor, prompt_post, device, weight_dtype,
                num_inference_steps=args.num_inference_steps,
            )

        # Create comparison: segmap_pre | generated_pre | real_pre | segmap_post | generated_post
        w, h = real_pre.size
        comparison = Image.new("RGB", (w * 5, h))
        comparison.paste(seg_pre_rgb_img.resize((w, h)), (0, 0))
        comparison.paste(gen_pre.resize((w, h)), (w, 0))
        comparison.paste(real_pre, (w * 2, 0))
        comparison.paste(seg_post_rgb_img.resize((w, h)), (w * 3, 0))
        comparison.paste(gen_post.resize((w, h)), (w * 4, 0))
        comparison.save(os.path.join(args.output_dir, f"{stem}_comparison.png"))

        gen_pre.save(os.path.join(args.output_dir, f"{stem}_gen_pre.png"))
        gen_post.save(os.path.join(args.output_dir, f"{stem}_gen_post.png"))

    print(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
