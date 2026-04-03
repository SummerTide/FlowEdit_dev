# rs_inference.py
"""Batch FlowEdit inference for RS change generation on Hi-UCD test set.

Reads bi-temporal pairs from Hi-UCD, runs FlowEdit with ControlNet to generate
post-temporal images, saves results for evaluation.

Usage:
    python rs_inference.py \
        --hiucd_root /path/to/hiucd \
        --controlnet_path ./checkpoints/sd3_controlnet_rs/controlnet_final \
        --output_dir ./outputs/rs_flowedit \
        --device_number 0
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import SD3ControlNetModel, StableDiffusion3Pipeline
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from FlowEdit_utils import FlowEditSD3ControlNet
from rs_data.hiucd import parse_hiucd_mask, hiucd_segmap_to_rgb, hiucd_segmap_to_text


def load_and_preprocess_image(pipe, image_path, device):
    """Load image, encode to VAE latent."""
    image = Image.open(image_path).convert("RGB")
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_tensor = pipe.image_processor.preprocess(image)
    image_tensor = image_tensor.to(device).half()
    with torch.autocast("cuda"), torch.inference_mode():
        x0_denorm = pipe.vae.encode(image_tensor).latent_dist.mode()
    x0 = (x0_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return x0.to(device)


def load_segmap_as_cond(segmap_path, resolution, device):
    """Load RGB segmap and prepare as ControlNet condition tensor."""
    seg_img = Image.open(segmap_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    seg_tensor = transform(seg_img).unsqueeze(0).to(device).half()
    return seg_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiucd_root", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/rs_flowedit")
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--T_steps", type=int, default=50)
    parser.add_argument("--n_avg", type=int, default=3)
    parser.add_argument("--src_guidance_scale", type=float, default=3.5)
    parser.add_argument("--tar_guidance_scale", type=float, default=13.5)
    parser.add_argument("--n_min", type=int, default=0)
    parser.add_argument("--n_max", type=int, default=35)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load models
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to(device)
    scheduler = pipe.scheduler

    controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16).to(device)
    controlnet.eval()

    # Iterate over Hi-UCD pairs
    # Structure: {split}/image/2018/, image/2019/, mask/2018_2019/
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

        # Load pre image latent
        x0_src = load_and_preprocess_image(pipe, str(pre_img_path), device)

        # Parse change mask into pre/post segmentation maps
        mask_rgb = np.array(Image.open(mask_path))
        pre_seg_np, post_seg_np, _change_label = parse_hiucd_mask(mask_rgb)

        pre_seg_rgb = hiucd_segmap_to_rgb(pre_seg_np)
        post_seg_rgb = hiucd_segmap_to_rgb(post_seg_np)

        # Save RGB segmaps for loading as tensors
        pre_seg_rgb_path = os.path.join(args.output_dir, f"{stem}_seg_pre.png")
        post_seg_rgb_path = os.path.join(args.output_dir, f"{stem}_seg_post.png")
        Image.fromarray(pre_seg_rgb).save(pre_seg_rgb_path)
        Image.fromarray(post_seg_rgb).save(post_seg_rgb_path)

        resolution = min(x0_src.shape[2] * 8, x0_src.shape[3] * 8)
        seg_src_cond = load_segmap_as_cond(pre_seg_rgb_path, resolution, device)
        seg_tar_cond = load_segmap_as_cond(post_seg_rgb_path, resolution, device)

        # Generate text prompts
        text_pre = hiucd_segmap_to_text(pre_seg_np)
        text_post = hiucd_segmap_to_text(post_seg_np)

        # Run FlowEdit
        x0_tar = FlowEditSD3ControlNet(
            pipe, scheduler, controlnet, x0_src,
            text_pre, text_post, "",
            seg_src_cond, seg_tar_cond,
            T_steps=args.T_steps,
            n_avg=args.n_avg,
            src_guidance_scale=args.src_guidance_scale,
            tar_guidance_scale=args.tar_guidance_scale,
            n_min=args.n_min,
            n_max=args.n_max,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        )

        # Decode
        x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
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
            "text_pre": text_pre,
            "text_post": text_post,
        })

    # Save results manifest
    manifest_path = os.path.join(args.output_dir, "results.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. {len(results)} images generated. Results: {manifest_path}")


if __name__ == "__main__":
    main()
