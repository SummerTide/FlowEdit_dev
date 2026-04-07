# rs_visualize_steps.py
"""Visualize intermediate FlowEdit steps from pre-image to post-image.

Runs FlowEdit+ControlNet on selected Hi-UCD samples, decodes the latent at
each step, and saves:
  - Individual step images
  - A combined strip (pre → intermediates → final | GT)
  - An animated GIF of the editing process

Usage:
    python rs_visualize_steps.py \
        --hiucd_root /path/to/Hi-UCD \
        --controlnet_path ./checkpoints/sd3_controlnet_rs_v7_1w/checkpoint-5500 \
        --output_dir ./outputs/rs_steps_vis \
        --split mini_val \
        --sample_indices 0 1 2 \
        --save_every 3
"""

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from diffusers import SD3ControlNetModel, StableDiffusion3Pipeline
from PIL import Image, ImageDraw, ImageFont
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
    device = image_tensor.device
    dtype = image_tensor.dtype
    with inference_autocast(device, dtype), torch.inference_mode():
        latent = pipe.vae.encode(image_tensor).latent_dist.mode()
    return (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


def decode_latent(pipe, latent, dtype):
    """Decode a latent tensor to a PIL Image."""
    x = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with inference_autocast(latent.device, dtype), torch.inference_mode():
        image = pipe.vae.decode(x, return_dict=False)[0]
    return pipe.image_processor.postprocess(image)[0]


def load_and_preprocess_image(pipe, image_path, device, dtype):
    image = Image.open(image_path).convert("RGB")
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    image_tensor = pipe.image_processor.preprocess(image).to(device=device, dtype=dtype)
    return encode_with_vae(pipe, image_tensor)


def load_segmap_as_controlnet_cond(pipe, segmap_rgb, resolution, device, dtype):
    seg_img = Image.fromarray(segmap_rgb).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    seg_tensor = transform(seg_img).unsqueeze(0).to(device=device, dtype=dtype)
    return encode_with_vae(pipe, seg_tensor)


def build_shared_prompt(pre_seg_np, post_seg_np, top_k=4):
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
    class_names = [HIUCD_CLASSES[c]["name"] for c in sorted_classes]
    if len(class_names) == 1:
        return f"an aerial remote sensing image with {class_names[0]}"
    return f"an aerial remote sensing image with {', '.join(class_names[:-1])} and {class_names[-1]}"


def add_label(img, text, position="bottom"):
    """Add a text label to the bottom or top of an image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if position == "bottom":
        xy = ((img.width - tw) // 2, img.height - th - 6)
    else:
        xy = ((img.width - tw) // 2, 4)
    # Draw shadow for readability
    draw.text((xy[0]+1, xy[1]+1), text, fill="black", font=font)
    draw.text(xy, text, fill="white", font=font)
    return img


def make_strip(images, labels=None, padding=4, bg_color=(40, 40, 40)):
    """Concatenate images horizontally with optional labels."""
    h = max(img.height for img in images)
    w_total = sum(img.width for img in images) + padding * (len(images) - 1)
    strip = Image.new("RGB", (w_total, h), bg_color)
    x = 0
    for i, img in enumerate(images):
        # Center vertically
        y = (h - img.height) // 2
        strip.paste(img, (x, y))
        if labels and i < len(labels):
            add_label(strip.crop((x, y, x + img.width, y + img.height)), labels[i])
            # Re-paste labeled image
            labeled = img.copy()
            add_label(labeled, labels[i])
            strip.paste(labeled, (x, y))
        x += img.width + padding
    return strip


def make_gif(frames, output_path, duration_ms=300, loop=0):
    """Save list of PIL Images as animated GIF."""
    if len(frames) < 2:
        return
    # Hold first and last frames longer
    durations = [duration_ms * 3] + [duration_ms] * (len(frames) - 2) + [duration_ms * 5]
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:],
        duration=durations, loop=loop,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize FlowEdit intermediate steps")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--hiucd_root", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/rs_steps_vis")
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--split", type=str, default="mini_val")

    # Sample selection
    parser.add_argument("--sample_indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Which samples (by sorted index) to visualize")

    # FlowEdit params
    parser.add_argument("--T_steps", type=int, default=50)
    parser.add_argument("--n_avg", type=int, default=3)
    parser.add_argument("--src_guidance_scale", type=float, default=3.5)
    parser.add_argument("--tar_guidance_scale", type=float, default=13.5)
    parser.add_argument("--n_min", type=int, default=8)
    parser.add_argument("--n_max", type=int, default=45)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=2.0)
    parser.add_argument("--v_delta_scale", type=float, default=1.0)
    parser.add_argument("--prompt_mode", type=str, default="shared_union",
                        choices=["paired", "shared_union"])
    parser.add_argument("--seed", type=int, default=42)

    # Visualization params
    parser.add_argument("--save_every", type=int, default=3,
                        help="Save an intermediate frame every N active steps")
    parser.add_argument("--gif_duration", type=int, default=300,
                        help="Duration per frame in GIF (ms)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load models
    print("Loading models...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype,
    ).to(device)
    scheduler = pipe.scheduler

    controlnet = SD3ControlNetModel.from_pretrained(
        args.controlnet_path, torch_dtype=weight_dtype,
    ).to(device)
    controlnet.eval()

    # Collect mask files
    split_dir = Path(args.hiucd_root) / args.split
    pre_img_dir = split_dir / "image" / "2018"
    post_img_dir = split_dir / "image" / "2019"
    mask_dir = split_dir / "mask" / "2018_2019"
    mask_files = sorted(mask_dir.glob("*.png"))

    os.makedirs(args.output_dir, exist_ok=True)

    for sample_idx in args.sample_indices:
        if sample_idx >= len(mask_files):
            print(f"Sample index {sample_idx} out of range (total {len(mask_files)}), skipping")
            continue

        mask_path = mask_files[sample_idx]
        stem = mask_path.stem
        print(f"\n--- Processing sample {sample_idx}: {stem} ---")

        pre_img_path = pre_img_dir / f"{stem}.png"
        post_img_path = post_img_dir / f"{stem}.png"

        if not pre_img_path.exists() or not post_img_path.exists():
            print(f"Skipping {stem}: missing image files")
            continue

        # Encode source image
        x0_src = load_and_preprocess_image(pipe, str(pre_img_path), device, weight_dtype)

        # Parse segmaps
        mask_rgb = np.array(Image.open(mask_path))
        pre_seg_np, post_seg_np, _ = parse_hiucd_mask(mask_rgb)
        pre_seg_rgb = hiucd_segmap_to_rgb(pre_seg_np)
        post_seg_rgb = hiucd_segmap_to_rgb(post_seg_np)

        vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
        resolution = min(x0_src.shape[2] * vae_scale_factor, x0_src.shape[3] * vae_scale_factor)
        seg_src_cond = load_segmap_as_controlnet_cond(pipe, pre_seg_rgb, resolution, device, weight_dtype)
        seg_tar_cond = load_segmap_as_controlnet_cond(pipe, post_seg_rgb, resolution, device, weight_dtype)

        # Build prompts
        if args.prompt_mode == "shared_union":
            shared = build_shared_prompt(pre_seg_np, post_seg_np)
            src_prompt, tar_prompt = shared, shared
        else:
            src_prompt = hiucd_segmap_to_text(pre_seg_np)
            tar_prompt = hiucd_segmap_to_text(post_seg_np)

        # Collect intermediate steps
        intermediates = []  # list of (step_idx, phase, pil_image)
        active_step_count = [0]  # mutable counter for closure

        def step_callback(step_idx, timestep, latent, phase):
            active_step_count[0] += 1
            if active_step_count[0] % args.save_every == 0 or active_step_count[0] == 1:
                img = decode_latent(pipe, latent, weight_dtype)
                intermediates.append((step_idx, phase, img))

        # Run FlowEdit
        print(f"Running FlowEdit (T={args.T_steps}, n_min={args.n_min}, n_max={args.n_max})...")
        x0_tar = FlowEditSD3ControlNet(
            pipe, scheduler, controlnet, x0_src,
            src_prompt, tar_prompt, "",
            seg_src_cond, seg_tar_cond,
            T_steps=args.T_steps,
            n_avg=args.n_avg,
            src_guidance_scale=args.src_guidance_scale,
            tar_guidance_scale=args.tar_guidance_scale,
            n_min=args.n_min,
            n_max=args.n_max,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            v_delta_scale=args.v_delta_scale,
            step_callback=step_callback,
        )

        # Decode final result
        final_img = decode_latent(pipe, x0_tar, weight_dtype)

        # Load reference images
        pre_img_pil = Image.open(pre_img_path).convert("RGB")
        post_img_pil = Image.open(post_img_path).convert("RGB")
        pre_seg_pil = Image.fromarray(pre_seg_rgb)
        post_seg_pil = Image.fromarray(post_seg_rgb)

        # Resize all to same size for strip
        target_size = pre_img_pil.size
        final_img = final_img.resize(target_size, Image.LANCZOS)

        # --- Save outputs ---
        sample_dir = os.path.join(args.output_dir, stem)
        os.makedirs(sample_dir, exist_ok=True)

        # 1. Individual step images
        for step_idx, phase, img in intermediates:
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(os.path.join(sample_dir, f"step_{step_idx:03d}_{phase}.png"))

        final_img.save(os.path.join(sample_dir, "final.png"))

        # 2. Combined strip: pre_seg | pre_img | step_N | ... | final | post_img | post_seg
        strip_images = [pre_seg_pil, pre_img_pil]
        strip_labels = ["Pre Seg", "Pre Image"]

        for step_idx, phase, img in intermediates:
            strip_images.append(img.resize(target_size, Image.LANCZOS))
            strip_labels.append(f"Step {step_idx} ({phase[0].upper()})")

        strip_images.extend([final_img, post_img_pil, post_seg_pil])
        strip_labels.extend(["Final", "GT Post", "Post Seg"])

        strip = make_strip(strip_images, strip_labels)
        strip.save(os.path.join(sample_dir, "strip.png"))
        print(f"Strip saved: {len(strip_images)} panels, {strip.size}")

        # 3. Animated GIF: pre → intermediates → final
        gif_frames = [pre_img_pil.copy()]
        for _, _, img in intermediates:
            gif_frames.append(img.resize(target_size, Image.LANCZOS))
        gif_frames.append(final_img)

        gif_path = os.path.join(sample_dir, "animation.gif")
        make_gif(gif_frames, gif_path, duration_ms=args.gif_duration)
        print(f"GIF saved: {len(gif_frames)} frames → {gif_path}")

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
