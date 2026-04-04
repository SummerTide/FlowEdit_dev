"""Validate SD3 ControlNet checkpoint by generating images from segmentation maps.

Tests pure ControlNet generation quality (from noise, not FlowEdit).
Generates images from Hi-UCD val set segmaps and saves results side by side.

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
from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline
from PIL import Image
from tqdm import tqdm

from rs_data.hiucd import parse_hiucd_mask, hiucd_segmap_to_rgb, hiucd_segmap_to_text


def main():
    parser = argparse.ArgumentParser(description="Validate SD3 ControlNet on Hi-UCD")
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to controlnet checkpoint")
    parser.add_argument("--hiucd_root", type=str, required=True, help="Path to Hi-UCD dataset root")
    parser.add_argument("--output_dir", type=str, default="./outputs/controlnet_val")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load pipeline
    print("Loading ControlNet checkpoint...")
    controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload(gpu_id=args.device_number)

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

        # Load real images for comparison
        real_pre = Image.open(os.path.join(pre_img_dir, mask_name)).convert("RGB")

        # Generate from pre segmap
        seg_pre_rgb = Image.fromarray(hiucd_segmap_to_rgb(seg_pre))
        prompt_pre = hiucd_segmap_to_text(seg_pre)

        with torch.autocast("cuda"):
            gen_pre = pipe(
                prompt=prompt_pre,
                control_image=seg_pre_rgb,
                num_inference_steps=args.num_inference_steps,
            ).images[0]

        # Generate from post segmap
        seg_post_rgb = Image.fromarray(hiucd_segmap_to_rgb(seg_post))
        prompt_post = hiucd_segmap_to_text(seg_post)

        with torch.autocast("cuda"):
            gen_post = pipe(
                prompt=prompt_post,
                control_image=seg_post_rgb,
                num_inference_steps=args.num_inference_steps,
            ).images[0]

        # Create comparison: segmap_pre | generated_pre | real_pre | segmap_post | generated_post
        w, h = real_pre.size
        comparison = Image.new("RGB", (w * 5, h))
        comparison.paste(seg_pre_rgb.resize((w, h)), (0, 0))
        comparison.paste(gen_pre.resize((w, h)), (w, 0))
        comparison.paste(real_pre, (w * 2, 0))
        comparison.paste(seg_post_rgb.resize((w, h)), (w * 3, 0))
        comparison.paste(gen_post.resize((w, h)), (w * 4, 0))
        comparison.save(os.path.join(args.output_dir, f"{stem}_comparison.png"))

        # Also save individual outputs
        gen_pre.save(os.path.join(args.output_dir, f"{stem}_gen_pre.png"))
        gen_post.save(os.path.join(args.output_dir, f"{stem}_gen_post.png"))

    print(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
