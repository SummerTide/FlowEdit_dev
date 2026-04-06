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
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from diffusers import SD3ControlNetModel, StableDiffusion3Pipeline
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from FlowEdit_utils import FlowEditSD3ControlNet
from rs_data.hiucd import (
    HIUCD_CLASSES,
    HIUCD_UNLABELED_IDX,
    hiucd_segmap_to_rgb,
    hiucd_segmap_to_text,
    parse_hiucd_mask,
)


def inference_autocast(device, dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def encode_with_vae(pipe, image_tensor):
    """Encode a normalized image tensor into the SD3 VAE latent space."""
    device = image_tensor.device
    dtype = image_tensor.dtype
    with inference_autocast(device, dtype), torch.inference_mode():
        latent = pipe.vae.encode(image_tensor).latent_dist.mode()
    return (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


def load_and_preprocess_image(pipe, image_path, device, dtype):
    """Load an RGB image and encode it to VAE latent."""
    image = Image.open(image_path).convert("RGB")
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_tensor = pipe.image_processor.preprocess(image).to(device=device, dtype=dtype)
    return encode_with_vae(pipe, image_tensor).to(device=device, dtype=dtype)


def load_segmap_as_controlnet_cond(pipe, segmap_rgb, resolution, device, dtype):
    """Load an RGB segmap and encode it into the ControlNet conditioning latent.

    `train_controlnet_sd3_rs.py` trains SD3 ControlNet on VAE latents derived
    from normalized RGB segmaps, so inference must mirror that preprocessing.
    """
    seg_img = Image.fromarray(segmap_rgb).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    seg_tensor = transform(seg_img).unsqueeze(0).to(device=device, dtype=dtype)
    return encode_with_vae(pipe, seg_tensor).to(device=device, dtype=dtype)


def format_class_prompt(class_names):
    if not class_names:
        return "an aerial remote sensing image"
    if len(class_names) == 1:
        return f"an aerial remote sensing image with {class_names[0]}"
    classes_str = ", ".join(class_names[:-1]) + " and " + class_names[-1]
    return f"an aerial remote sensing image with {classes_str}"


def build_shared_segmap_prompt(pre_seg_np, post_seg_np, top_k=4):
    """Build one shared prompt so edits are driven by segmap changes, not text deltas."""
    merged_counts = {}
    for class_idx, info in HIUCD_CLASSES.items():
        if class_idx == HIUCD_UNLABELED_IDX:
            continue
        count = int(np.sum(pre_seg_np == class_idx) + np.sum(post_seg_np == class_idx))
        if count > 0:
            merged_counts[class_idx] = count

    if not merged_counts:
        return "an aerial remote sensing image"

    sorted_classes = sorted(merged_counts, key=merged_counts.get, reverse=True)[:top_k]
    class_names = [HIUCD_CLASSES[class_idx]["name"] for class_idx in sorted_classes]
    return format_class_prompt(class_names)


def resolve_flowedit_prompts(pre_seg_np, post_seg_np, prompt_mode, shared_prompt_top_k, neutral_prompt):
    text_pre = hiucd_segmap_to_text(pre_seg_np)
    text_post = hiucd_segmap_to_text(post_seg_np)

    if prompt_mode == "paired":
        return text_pre, text_post, text_pre, text_post
    if prompt_mode == "shared_union":
        shared_prompt = build_shared_segmap_prompt(pre_seg_np, post_seg_np, top_k=shared_prompt_top_k)
        return text_pre, text_post, shared_prompt, shared_prompt
    if prompt_mode == "neutral":
        return text_pre, text_post, neutral_prompt, neutral_prompt

    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
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
    parser.add_argument("--v_delta_scale", type=float, default=1.0, help="Scale factor for the FlowEdit velocity delta (V_tar - V_src). Values > 1 amplify segmap-driven changes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="shared_union",
        choices=["paired", "shared_union", "neutral"],
        help="How text prompts are used. Shared modes keep source/target text identical so edits are mainly driven by segmap differences.",
    )
    parser.add_argument("--shared_prompt_top_k", type=int, default=4)
    parser.add_argument("--neutral_prompt", type=str, default="an aerial remote sensing image")
    parser.add_argument("--equalize_guidance", action="store_true", help="Force tar_guidance_scale = src_guidance_scale in shared/neutral modes. Off by default so ControlNet differences are amplified by higher tar guidance.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load models
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
    ).to(device)
    scheduler = pipe.scheduler

    controlnet = SD3ControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=weight_dtype,
    ).to(device)
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
        x0_src = load_and_preprocess_image(pipe, str(pre_img_path), device, weight_dtype)

        # Parse change mask into pre/post segmentation maps
        mask_rgb = np.array(Image.open(mask_path))
        pre_seg_np, post_seg_np, _change_label = parse_hiucd_mask(mask_rgb)

        pre_seg_rgb = hiucd_segmap_to_rgb(pre_seg_np)
        post_seg_rgb = hiucd_segmap_to_rgb(post_seg_np)

        vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
        resolution = min(x0_src.shape[2] * vae_scale_factor, x0_src.shape[3] * vae_scale_factor)
        seg_src_cond = load_segmap_as_controlnet_cond(pipe, pre_seg_rgb, resolution, device, weight_dtype)
        seg_tar_cond = load_segmap_as_controlnet_cond(pipe, post_seg_rgb, resolution, device, weight_dtype)

        # Resolve text conditioning. Shared modes intentionally remove text deltas
        # so the edit is primarily driven by segmap / ControlNet differences.
        text_pre, text_post, flowedit_src_prompt, flowedit_tar_prompt = resolve_flowedit_prompts(
            pre_seg_np,
            post_seg_np,
            args.prompt_mode,
            args.shared_prompt_top_k,
            args.neutral_prompt,
        )

        effective_tar_guidance_scale = args.tar_guidance_scale
        if args.prompt_mode != "paired" and args.equalize_guidance:
            effective_tar_guidance_scale = args.src_guidance_scale

        # Run FlowEdit
        x0_tar = FlowEditSD3ControlNet(
            pipe, scheduler, controlnet, x0_src,
            flowedit_src_prompt, flowedit_tar_prompt, "",
            seg_src_cond, seg_tar_cond,
            T_steps=args.T_steps,
            n_avg=args.n_avg,
            src_guidance_scale=args.src_guidance_scale,
            tar_guidance_scale=effective_tar_guidance_scale,
            n_min=args.n_min,
            n_max=args.n_max,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            v_delta_scale=args.v_delta_scale,
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
            "seg_text_pre": text_pre,
            "seg_text_post": text_post,
            "flowedit_src_prompt": flowedit_src_prompt,
            "flowedit_tar_prompt": flowedit_tar_prompt,
            "prompt_mode": args.prompt_mode,
        })

    # Save results manifest
    manifest_path = os.path.join(args.output_dir, "results.csv")
    if results:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"Done. {len(results)} images generated. Results: {manifest_path}")


if __name__ == "__main__":
    main()
