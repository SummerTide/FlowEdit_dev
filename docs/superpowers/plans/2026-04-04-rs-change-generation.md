# RS Change Generation via FlowEdit + ControlNet — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt FlowEdit to generate post-temporal remote sensing images from pre-temporal images, conditioned on semantic segmentation maps via SD3 ControlNet, for change detection data augmentation.

**Architecture:** SD3 (frozen) + ControlNet (trainable on single-temporal RS segmap→image pairs) + FlowEdit V_delta ODE. ControlNet provides semantic map conditioning; FlowEdit drives the direct pre→post transport without inversion.

**Tech Stack:** Python, PyTorch, diffusers (0.30.1), accelerate, PIL, numpy, yaml, torchmetrics (FID/LPIPS)

**Spec:** `docs/superpowers/specs/2026-04-04-rs-change-generation-design.md`

---

## File Structure

```
FlowEdit_dev/
├── FlowEdit_utils.py              # MODIFY: add ControlNet-conditioned calc_v_sd3_controlnet(), FlowEditSD3ControlNet()
├── run_script.py                   # MODIFY: add SD3_ControlNet model_type branch
├── rs_data/                        # CREATE: data preparation utilities
│   ├── prepare_hiucd.py            # CREATE: Hi-UCD single-temporal split + text prompt generation
│   └── rs_dataset.py               # CREATE: PyTorch Dataset for ControlNet training
├── train_controlnet_sd3_rs.py      # CREATE: ControlNet training script (adapted from diffusers example)
├── rs_inference.py                 # CREATE: batch FlowEdit inference on Hi-UCD test set
├── rs_evaluate.py                  # CREATE: FID, LPIPS, SSIM, PSNR evaluation
├── configs/                        # CREATE: RS experiment configs
│   ├── hiucd_edits.yaml            # CREATE: Hi-UCD test set edit definitions
│   └── RS_SD3_exp.yaml             # CREATE: RS FlowEdit experiment hyperparameters
└── rs_data/class_mapping.py        # CREATE: Hi-UCD class index → RGB + text label mapping
```

---

## Task 1: Hi-UCD Class Mapping and Text Generation Utilities

**Files:**
- Create: `rs_data/__init__.py`
- Create: `rs_data/class_mapping.py`

- [ ] **Step 1: Create rs_data package init**

```python
# rs_data/__init__.py
# (empty)
```

- [ ] **Step 2: Create class mapping module**

Hi-UCD has 9 classes. Create the mapping from class index to RGB color and text label.

```python
# rs_data/class_mapping.py
"""Hi-UCD dataset class mapping: index → RGB color and text label."""

import numpy as np
from PIL import Image

# Hi-UCD 9-class mapping
# Classes: 0-Unchanged, 1-Water, 2-Ground/Barren, 3-Low Vegetation,
#          4-Tree, 5-Building, 6-Playground, 7-Road, 8-Others
HIUCD_CLASSES = {
    0: {"name": "unchanged area", "color": (0, 0, 0)},
    1: {"name": "water", "color": (0, 0, 255)},
    2: {"name": "barren ground", "color": (128, 128, 128)},
    3: {"name": "low vegetation", "color": (0, 255, 0)},
    4: {"name": "trees", "color": (0, 128, 0)},
    5: {"name": "buildings", "color": (255, 0, 0)},
    6: {"name": "playground", "color": (255, 255, 0)},
    7: {"name": "roads", "color": (255, 128, 0)},
    8: {"name": "other structures", "color": (128, 0, 128)},
}


def segmap_to_rgb(segmap_np: np.ndarray) -> np.ndarray:
    """Convert class-index segmentation map (H, W) to RGB image (H, W, 3).

    Args:
        segmap_np: numpy array of shape (H, W) with integer class indices.

    Returns:
        RGB numpy array of shape (H, W, 3), dtype uint8.
    """
    h, w = segmap_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, info in HIUCD_CLASSES.items():
        mask = segmap_np == class_idx
        rgb[mask] = info["color"]
    return rgb


def segmap_to_text(segmap_np: np.ndarray, top_k: int = 3) -> str:
    """Generate a text prompt from a segmentation map based on class area ratios.

    Args:
        segmap_np: numpy array of shape (H, W) with integer class indices.
        top_k: number of top classes (by area) to include in the prompt.

    Returns:
        Text prompt string, e.g. "an aerial remote sensing image with buildings, roads and trees"
    """
    total_pixels = segmap_np.size
    class_counts = {}
    for class_idx, info in HIUCD_CLASSES.items():
        if class_idx == 0:  # skip "unchanged area" — not meaningful for single-temporal description
            continue
        count = np.sum(segmap_np == class_idx)
        if count > 0:
            class_counts[class_idx] = count

    if not class_counts:
        return "an aerial remote sensing image"

    # Sort by area descending, take top_k
    sorted_classes = sorted(class_counts.keys(), key=lambda k: class_counts[k], reverse=True)[:top_k]
    class_names = [HIUCD_CLASSES[c]["name"] for c in sorted_classes]

    if len(class_names) == 1:
        classes_str = class_names[0]
    else:
        classes_str = ", ".join(class_names[:-1]) + " and " + class_names[-1]

    return f"an aerial remote sensing image with {classes_str}"
```

- [ ] **Step 3: Verify the module loads**

Run: `cd /home/summertide/Workspace/FlowEdit_dev && python -c "from rs_data.class_mapping import segmap_to_rgb, segmap_to_text, HIUCD_CLASSES; print(f'{len(HIUCD_CLASSES)} classes loaded'); import numpy as np; seg = np.random.randint(0, 9, (64, 64)); print(segmap_to_text(seg)); print(segmap_to_rgb(seg).shape)"`

Expected: `9 classes loaded`, a text prompt, and `(64, 64, 3)`.

- [ ] **Step 4: Commit**

```bash
git add rs_data/__init__.py rs_data/class_mapping.py
git commit -m "feat: add Hi-UCD class mapping and text prompt generation utilities"
```

---

## Task 2: Hi-UCD Data Preparation Script

**Files:**
- Create: `rs_data/prepare_hiucd.py`

This script takes the raw Hi-UCD dataset and produces the single-temporal training split.

- [ ] **Step 1: Create data preparation script**

```python
# rs_data/prepare_hiucd.py
"""Prepare Hi-UCD dataset for ControlNet training.

Splits bi-temporal pairs into single-temporal (image, segmap_rgb, text) samples.
Outputs a training manifest CSV.

Expected Hi-UCD directory structure:
    hiucd_root/
    ├── train/
    │   ├── image1/          # pre-temporal image
    │   ├── image2/          # post-temporal image
    │   ├── label1/          # pre-temporal segmentation
    │   └── label2/          # post-temporal segmentation
    └── test/
        └── ...

Usage:
    python -m rs_data.prepare_hiucd --hiucd_root /path/to/hiucd --output_dir ./data/hiucd_prepared
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

from rs_data.class_mapping import segmap_to_rgb, segmap_to_text


def prepare_split(hiucd_root: str, split: str, output_dir: str) -> list:
    """Process one split (train/test) of Hi-UCD.

    Returns list of dicts: {image_path, segmap_rgb_path, text_prompt}
    """
    split_dir = Path(hiucd_root) / split
    out_dir = Path(output_dir) / split
    out_img_dir = out_dir / "images"
    out_seg_dir = out_dir / "segmaps"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # Process both temporal phases
    for phase, img_folder, label_folder in [("pre", "image1", "label1"), ("post", "image2", "label2")]:
        img_dir = split_dir / img_folder
        label_dir = split_dir / label_folder

        if not img_dir.exists():
            print(f"Warning: {img_dir} not found, skipping")
            continue

        img_files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.jpg"))

        for img_path in img_files:
            stem = img_path.stem
            # Find matching label file
            label_path = None
            for ext in [".png", ".tif", ".jpg"]:
                candidate = label_dir / f"{stem}{ext}"
                if candidate.exists():
                    label_path = candidate
                    break

            if label_path is None:
                print(f"Warning: no label found for {img_path}, skipping")
                continue

            # Load and process
            image = Image.open(img_path).convert("RGB")
            segmap = np.array(Image.open(label_path))

            # Generate RGB segmap and text prompt
            segmap_rgb = segmap_to_rgb(segmap)
            text_prompt = segmap_to_text(segmap)

            # Save processed files
            out_name = f"{phase}_{stem}"
            out_img_path = out_img_dir / f"{out_name}.png"
            out_seg_path = out_seg_dir / f"{out_name}.png"

            image.save(out_img_path)
            Image.fromarray(segmap_rgb).save(out_seg_path)

            records.append({
                "image_path": str(out_img_path),
                "segmap_path": str(out_seg_path),
                "text_prompt": text_prompt,
                "phase": phase,
                "original_name": stem,
            })

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare Hi-UCD for ControlNet training")
    parser.add_argument("--hiucd_root", type=str, required=True, help="Path to Hi-UCD dataset root")
    parser.add_argument("--output_dir", type=str, default="./data/hiucd_prepared", help="Output directory")
    args = parser.parse_args()

    all_records = []
    for split in ["train", "test"]:
        records = prepare_split(args.hiucd_root, split, args.output_dir)
        all_records.extend(records)
        print(f"{split}: {len(records)} samples prepared")

    # Write manifest CSV
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "segmap_path", "text_prompt", "phase", "original_name"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Total: {len(all_records)} samples. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses without errors**

Run: `cd /home/summertide/Workspace/FlowEdit_dev && python -c "from rs_data.prepare_hiucd import prepare_split; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rs_data/prepare_hiucd.py
git commit -m "feat: add Hi-UCD data preparation script for single-temporal split"
```

---

## Task 3: PyTorch Dataset for ControlNet Training

**Files:**
- Create: `rs_data/rs_dataset.py`

- [ ] **Step 1: Create the dataset class**

```python
# rs_data/rs_dataset.py
"""PyTorch Dataset for SD3 ControlNet training on remote sensing data."""

import csv
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RSControlNetDataset(Dataset):
    """Dataset that loads (image, segmap_rgb, text_prompt) triplets from a manifest CSV.

    CSV columns: image_path, segmap_path, text_prompt, phase, original_name
    """

    def __init__(self, manifest_path: str, resolution: int = 1024, split: str = "train"):
        self.resolution = resolution
        self.samples = []

        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by split: train samples come from "train/" in the path
                if split in row["image_path"]:
                    self.samples.append(row)

        self.image_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # scale to [-1, 1]
        ])

        self.seg_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

        # Data augmentation (applied to both image and segmap consistently)
        self.augment = split == "train"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        segmap = Image.open(row["segmap_path"]).convert("RGB")

        # Consistent random augmentation for image and segmap
        if self.augment:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                segmap = segmap.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                segmap = segmap.transpose(Image.FLIP_TOP_BOTTOM)
            rot_choice = random.choice([0, 90, 180, 270])
            if rot_choice > 0:
                image = image.rotate(rot_choice, expand=False)
                segmap = segmap.rotate(rot_choice, expand=False)

        image_tensor = self.image_transform(image)
        seg_tensor = self.seg_transform(segmap)

        return {
            "pixel_values": image_tensor,
            "conditioning_pixel_values": seg_tensor,
            "caption": row["text_prompt"],
        }
```

- [ ] **Step 2: Verify dataset class loads**

Run: `cd /home/summertide/Workspace/FlowEdit_dev && python -c "from rs_data.rs_dataset import RSControlNetDataset; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rs_data/rs_dataset.py
git commit -m "feat: add PyTorch dataset for RS ControlNet training"
```

---

## Task 4: ControlNet Training Script

**Files:**
- Create: `train_controlnet_sd3_rs.py`

This is adapted from diffusers' official `train_controlnet_sd3.py` but simplified for our use case.

- [ ] **Step 1: Create the training script**

```python
# train_controlnet_sd3_rs.py
"""Train SD3 ControlNet on remote sensing segmentation maps.

Usage:
    accelerate launch train_controlnet_sd3_rs.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
        --manifest_path ./data/hiucd_prepared/manifest.csv \
        --output_dir ./checkpoints/sd3_controlnet_rs \
        --resolution 1024 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        --max_train_steps 15000 \
        --validation_steps 500 \
        --checkpointing_steps 1000
"""

import argparse
import logging
import math
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rs_data.rs_dataset import RSControlNetDataset

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SD3 ControlNet for RS segmap conditioning")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest.csv from prepare_hiucd.py")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sd3_controlnet_rs")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=15000)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_validation_images", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
    )
    logging.basicConfig(level=logging.INFO)

    # Load base pipeline components
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
    )
    transformer = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    text_encoder_3 = pipe.text_encoder_3
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    tokenizer_3 = pipe.tokenizer_3
    scheduler = pipe.scheduler

    # Initialize ControlNet from transformer
    controlnet = SD3ControlNetModel.from_transformer(transformer)

    # Freeze everything except controlnet
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)
    if text_encoder_3 is not None:
        text_encoder_3.requires_grad_(False)

    controlnet.train()

    # Dataset and DataLoader
    dataset = RSControlNetDataset(args.manifest_path, resolution=args.resolution, split="train")
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with accelerator
    controlnet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, dataloader, lr_scheduler
    )

    # Move frozen models to device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_2 is not None:
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_3 is not None:
        text_encoder_3.to(accelerator.device, dtype=weight_dtype)

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), desc="Training", disable=not accelerator.is_local_main_process)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(controlnet):
                # Encode images to latent space
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                # Encode text
                captions = batch["caption"]
                with torch.no_grad():
                    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                        pipe.encode_prompt(
                            prompt=captions,
                            prompt_2=None,
                            prompt_3=None,
                            device=accelerator.device,
                            do_classifier_free_guidance=False,
                        )
                    )

                # Prepare conditioning image
                controlnet_cond = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample random timestep for each sample (flow matching uses uniform [0, 1])
                t = torch.rand(bsz, device=latents.device, dtype=weight_dtype)
                timestep = t * 1000  # SD3 uses timestep in [0, 1000]

                # Create noisy latents (linear interpolation for flow matching)
                sigmas = t.view(-1, 1, 1, 1)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise

                # Target velocity
                target = noise - latents

                # ControlNet forward
                controlnet_block_samples = controlnet(
                    hidden_states=noisy_latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )[0]

                # Transformer forward with ControlNet conditioning
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=controlnet_block_samples,
                    return_dict=False,
                )[0]

                # Flow matching loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(controlnet).save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

                # Validation
                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    logger.info(f"Running validation at step {global_step}...")
                    val_dataset = RSControlNetDataset(args.manifest_path, resolution=args.resolution, split="test")
                    if len(val_dataset) > 0:
                        val_pipe = StableDiffusion3ControlNetPipeline(
                            transformer=transformer,
                            controlnet=accelerator.unwrap_model(controlnet),
                            scheduler=scheduler,
                            vae=vae,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            text_encoder_3=text_encoder_3,
                            tokenizer=tokenizer,
                            tokenizer_2=tokenizer_2,
                            tokenizer_3=tokenizer_3,
                        )
                        val_dir = os.path.join(args.output_dir, f"validation-{global_step}")
                        os.makedirs(val_dir, exist_ok=True)
                        for vi in range(min(args.num_validation_images, len(val_dataset))):
                            sample = val_dataset[vi]
                            seg_img = Image.fromarray((sample["conditioning_pixel_values"].permute(1, 2, 0).numpy() * 255).astype("uint8"))
                            with torch.autocast("cuda"):
                                out = val_pipe(
                                    prompt=sample["caption"],
                                    control_image=seg_img,
                                    num_inference_steps=50,
                                ).images[0]
                            out.save(os.path.join(val_dir, f"val_{vi}.png"))
                            seg_img.save(os.path.join(val_dir, f"val_{vi}_seg.png"))
                        del val_pipe
                        torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet_unwrapped = accelerator.unwrap_model(controlnet)
        controlnet_unwrapped.save_pretrained(os.path.join(args.output_dir, "controlnet_final"))
        logger.info(f"Training complete. Final model saved to {args.output_dir}/controlnet_final")

    accelerator.end_training()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses**

Run: `cd /home/summertide/Workspace/FlowEdit_dev && python -c "import ast; ast.parse(open('train_controlnet_sd3_rs.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add train_controlnet_sd3_rs.py
git commit -m "feat: add SD3 ControlNet training script for RS segmap conditioning"
```

---

## Task 5: Adapt FlowEdit Velocity Calculation for ControlNet

**Files:**
- Modify: `FlowEdit_utils.py` — add `calc_v_sd3_controlnet()` function after existing `calc_v_sd3()` (line ~80)

- [ ] **Step 1: Add ControlNet-conditioned velocity function**

Add this function after `calc_v_sd3()` (after line 80) in `FlowEdit_utils.py`:

```python
def calc_v_sd3_controlnet(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t, controlnet, seg_src_cond, seg_tar_cond, controlnet_conditioning_scale=1.0):
    """Velocity calculation with ControlNet conditioning for SD3.

    Same as calc_v_sd3 but injects ControlNet residuals from segmentation maps.
    src uses seg_src_cond, tar uses seg_tar_cond.
    """
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
        # Build controlnet_cond: match the 4-way CFG batch order
        # [src_uncond, src_cond, tar_uncond, tar_cond]
        if pipe.do_classifier_free_guidance:
            controlnet_cond = torch.cat([seg_src_cond, seg_src_cond, seg_tar_cond, seg_tar_cond])
        else:
            controlnet_cond = torch.cat([seg_src_cond, seg_tar_cond])

        # ControlNet forward
        controlnet_block_samples = controlnet(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            controlnet_cond=controlnet_cond,
            conditioning_scale=controlnet_conditioning_scale,
            return_dict=False,
        )[0]

        # Transformer forward with ControlNet residuals
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            block_controlnet_hidden_states=controlnet_block_samples,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar
```

- [ ] **Step 2: Verify the file still parses**

Run: `cd /home/summertide/Workspace/FlowEdit_dev && python -c "import ast; ast.parse(open('FlowEdit_utils.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add FlowEdit_utils.py
git commit -m "feat: add calc_v_sd3_controlnet() for ControlNet-conditioned velocity"
```

---

## Task 6: Adapt FlowEdit ODE Loop for ControlNet

**Files:**
- Modify: `FlowEdit_utils.py` — add `FlowEditSD3ControlNet()` function after existing `FlowEditSD3()` (after line ~227)

- [ ] **Step 1: Add the ControlNet-aware FlowEdit function**

Add this after `FlowEditSD3()` in `FlowEdit_utils.py`:

```python
@torch.no_grad()
def FlowEditSD3ControlNet(pipe,
    scheduler,
    controlnet,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    seg_src_cond,
    seg_tar_cond,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 33,
    controlnet_conditioning_scale: float = 1.0,):
    """FlowEdit ODE with ControlNet conditioning on semantic segmentation maps.

    Same as FlowEditSD3 but passes seg_src_cond / seg_tar_cond through ControlNet
    when computing source / target velocities.

    Args:
        controlnet: trained SD3ControlNetModel instance
        seg_src_cond: pre-temporal segmap RGB, tensor (1, 3, H, W), values in [0, 1]
        seg_tar_cond: post-temporal segmap RGB, tensor (1, 3, H, W), values in [0, 1]
        controlnet_conditioning_scale: strength of ControlNet conditioning (default 1.0)
    """
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale

    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)

    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = (timesteps[i + 1]) / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)

                zt_src = (1 - t_i) * x_src + (t_i) * fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else torch.cat([zt_src, zt_tar])

                Vt_src, Vt_tar = calc_v_sd3_controlnet(
                    pipe, src_tar_latent_model_input,
                    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    src_guidance_scale, tar_guidance_scale, t,
                    controlnet, seg_src_cond, seg_tar_cond,
                    controlnet_conditioning_scale,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:  # regular sampling for last n_min steps

            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else xt_tar

            _, Vt_tar = calc_v_sd3_controlnet(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t,
                controlnet, seg_tar_cond, seg_tar_cond,
                controlnet_conditioning_scale,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    return zt_edit if n_min == 0 else xt_tar
```

- [ ] **Step 2: Verify parse**

Run: `cd /home/summertide/Workspace/FlowEdit_dev && python -c "import ast; ast.parse(open('FlowEdit_utils.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add FlowEdit_utils.py
git commit -m "feat: add FlowEditSD3ControlNet() for segmap-conditioned change generation"
```

---

## Task 7: RS Experiment Configuration Files

**Files:**
- Create: `configs/RS_SD3_exp.yaml`
- Create: `configs/hiucd_edits.yaml`

- [ ] **Step 1: Create experiment config**

```yaml
# configs/RS_SD3_exp.yaml
-
  exp_name: "FlowEdit_RS_SD3_ControlNet"
  dataset_yaml: configs/hiucd_edits.yaml
  model_type: "SD3_ControlNet"
  controlnet_path: "./checkpoints/sd3_controlnet_rs/controlnet_final"
  T_steps: 50
  n_avg: 3
  src_guidance_scale: 3.5
  tar_guidance_scale: 13.5
  n_min: 0
  n_max: 35
  controlnet_conditioning_scale: 1.0
  seed: 42
```

- [ ] **Step 2: Create a placeholder edits config (to be populated after data preparation)**

```yaml
# configs/hiucd_edits.yaml
# Each entry defines a change generation task.
# Populated by rs_inference.py from the Hi-UCD test set manifest.
# Format:
# - input_img: path/to/pre_image.png
#   source_prompt: "an aerial remote sensing image with ..."
#   target_prompts:
#     - "an aerial remote sensing image with ..."
#   seg_pre: path/to/seg_pre_rgb.png
#   seg_post: path/to/seg_post_rgb.png
```

- [ ] **Step 3: Commit**

```bash
mkdir -p configs
git add configs/RS_SD3_exp.yaml configs/hiucd_edits.yaml
git commit -m "feat: add RS experiment YAML configs"
```

---

## Task 8: RS Batch Inference Script

**Files:**
- Create: `rs_inference.py`

- [ ] **Step 1: Create the batch inference script**

```python
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
from rs_data.class_mapping import segmap_to_rgb, segmap_to_text


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

    # Iterate over Hi-UCD test pairs
    split_dir = Path(args.hiucd_root) / args.split
    pre_img_dir = split_dir / "image1"
    post_img_dir = split_dir / "image2"
    pre_label_dir = split_dir / "label1"
    post_label_dir = split_dir / "label2"

    pre_images = sorted(pre_img_dir.glob("*.png")) + sorted(pre_img_dir.glob("*.tif"))
    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    for pre_img_path in tqdm(pre_images, desc="Generating"):
        stem = pre_img_path.stem

        # Find corresponding files
        post_img_path = None
        pre_label_path = None
        post_label_path = None
        for ext in [".png", ".tif", ".jpg"]:
            if (post_img_dir / f"{stem}{ext}").exists():
                post_img_path = post_img_dir / f"{stem}{ext}"
            if (pre_label_dir / f"{stem}{ext}").exists():
                pre_label_path = pre_label_dir / f"{stem}{ext}"
            if (post_label_dir / f"{stem}{ext}").exists():
                post_label_path = post_label_dir / f"{stem}{ext}"

        if any(p is None for p in [post_img_path, pre_label_path, post_label_path]):
            print(f"Skipping {stem}: missing files")
            continue

        # Load pre image latent
        x0_src = load_and_preprocess_image(pipe, str(pre_img_path), device)

        # Load segmaps as RGB conditions
        pre_seg_np = np.array(Image.open(pre_label_path))
        post_seg_np = np.array(Image.open(post_label_path))

        pre_seg_rgb = segmap_to_rgb(pre_seg_np)
        post_seg_rgb = segmap_to_rgb(post_seg_np)

        # Save RGB segmaps temporarily for loading
        pre_seg_rgb_path = os.path.join(args.output_dir, f"{stem}_seg_pre.png")
        post_seg_rgb_path = os.path.join(args.output_dir, f"{stem}_seg_post.png")
        Image.fromarray(pre_seg_rgb).save(pre_seg_rgb_path)
        Image.fromarray(post_seg_rgb).save(post_seg_rgb_path)

        resolution = min(x0_src.shape[2] * 8, x0_src.shape[3] * 8)  # approx original resolution
        seg_src_cond = load_segmap_as_cond(pre_seg_rgb_path, resolution, device)
        seg_tar_cond = load_segmap_as_cond(post_seg_rgb_path, resolution, device)

        # Generate text prompts
        text_pre = segmap_to_text(pre_seg_np)
        text_post = segmap_to_text(post_seg_np)

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
```

- [ ] **Step 2: Verify parse**

Run: `cd /home/summertide/Workspace/FlowEdit_dev && python -c "import ast; ast.parse(open('rs_inference.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add rs_inference.py
git commit -m "feat: add batch RS FlowEdit inference script"
```

---

## Task 9: Evaluation Script

**Files:**
- Create: `rs_evaluate.py`

- [ ] **Step 1: Create evaluation script**

```python
# rs_evaluate.py
"""Evaluate generated RS images: FID, LPIPS, SSIM, PSNR.

Usage:
    python rs_evaluate.py \
        --results_csv ./outputs/rs_flowedit/results.csv \
        --output_dir ./outputs/rs_flowedit/eval
"""

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_image_tensor(path, size=None):
    """Load image as tensor in [0, 1] range."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return transforms.ToTensor()(img)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute PSNR between two image tensors in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM between two image tensors in [0, 1]. Simple per-channel mean."""
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()


def compute_lpips_score(img1: torch.Tensor, img2: torch.Tensor, lpips_fn) -> float:
    """Compute LPIPS between two image tensors in [0, 1]."""
    # LPIPS expects [-1, 1]
    return lpips_fn(img1.unsqueeze(0) * 2 - 1, img2.unsqueeze(0) * 2 - 1).item()


def compute_fid(real_dir: str, gen_dir: str, device: torch.device) -> float:
    """Compute FID between two directories of images."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True).to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Load real images
    for fname in sorted(os.listdir(real_dir)):
        if fname.lower().endswith((".png", ".jpg", ".tif")):
            img = transform(Image.open(os.path.join(real_dir, fname)).convert("RGB"))
            fid.update(img.unsqueeze(0).to(device), real=True)

    # Load generated images
    for fname in sorted(os.listdir(gen_dir)):
        if fname.lower().endswith((".png", ".jpg", ".tif")):
            img = transform(Image.open(os.path.join(gen_dir, fname)).convert("RGB"))
            fid.update(img.unsqueeze(0).to(device), real=False)

    return fid.compute().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True, help="Path to results.csv from rs_inference.py")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--compute_fid", action="store_true", help="Compute FID (requires many samples)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_csv)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load results
    with open(args.results_csv, "r") as f:
        results = list(csv.DictReader(f))

    print(f"Evaluating {len(results)} pairs...")

    # Per-pair metrics
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    # Lazy load LPIPS
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    except ImportError:
        print("Warning: torchmetrics not available, skipping LPIPS")
        lpips_fn = None

    for row in results:
        real_img = load_image_tensor(row["post_img_real"], size=1024).to(device)
        gen_img = load_image_tensor(row["post_img_generated"], size=1024).to(device)

        psnr_scores.append(compute_psnr(gen_img, real_img))
        ssim_scores.append(compute_ssim(gen_img.cpu(), real_img.cpu()))

        if lpips_fn is not None:
            lpips_scores.append(compute_lpips_score(gen_img, real_img, lpips_fn))

    # Aggregate
    report_lines = [
        f"Evaluation Report ({len(results)} pairs)",
        f"{'='*40}",
        f"PSNR:  {np.mean(psnr_scores):.2f} +/- {np.std(psnr_scores):.2f}",
        f"SSIM:  {np.mean(ssim_scores):.4f} +/- {np.std(ssim_scores):.4f}",
    ]

    if lpips_scores:
        report_lines.append(f"LPIPS: {np.mean(lpips_scores):.4f} +/- {np.std(lpips_scores):.4f}")

    # FID (optional, needs enough samples)
    if args.compute_fid:
        # Prepare temp dirs with symlinks
        real_tmp = os.path.join(args.output_dir, "fid_real")
        gen_tmp = os.path.join(args.output_dir, "fid_gen")
        os.makedirs(real_tmp, exist_ok=True)
        os.makedirs(gen_tmp, exist_ok=True)

        for row in results:
            stem = row["stem"]
            real_src = row["post_img_real"]
            gen_src = row["post_img_generated"]
            real_dst = os.path.join(real_tmp, f"{stem}.png")
            gen_dst = os.path.join(gen_tmp, f"{stem}.png")
            if not os.path.exists(real_dst):
                os.symlink(os.path.abspath(real_src), real_dst)
            if not os.path.exists(gen_dst):
                os.symlink(os.path.abspath(gen_src), gen_dst)

        fid_score = compute_fid(real_tmp, gen_tmp, device)
        report_lines.append(f"FID:   {fid_score:.2f}")

    report = "\n".join(report_lines)
    print(report)

    # Save report
    report_path = os.path.join(args.output_dir, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify parse**

Run: `cd /home/summertide/Workspace/FlowEdit_dev && python -c "import ast; ast.parse(open('rs_evaluate.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add rs_evaluate.py
git commit -m "feat: add evaluation script for RS change generation (FID, LPIPS, SSIM, PSNR)"
```

---

## Task 10: Update run_script.py for SD3_ControlNet Model Type

**Files:**
- Modify: `run_script.py` — add `SD3_ControlNet` branch

- [ ] **Step 1: Add SD3_ControlNet imports and model loading**

At the top of `run_script.py`, add to the imports:

```python
from diffusers import SD3ControlNetModel
from FlowEdit_utils import FlowEditSD3, FlowEditFLUX, FlowEditSD3ControlNet
```

After the existing SD3 pipeline loading block (around line 38), add:

```python
    elif model_type == 'SD3_ControlNet':
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        controlnet_path = exp_configs[0].get("controlnet_path", "./checkpoints/sd3_controlnet_rs/controlnet_final")
        controlnet = SD3ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
```

- [ ] **Step 2: Add SD3_ControlNet inference branch in the main loop**

In the inner loop where model_type is checked (around line 92), add a new branch before the `else` clause:

```python
                elif model_type == 'SD3_ControlNet':
                    from PIL import Image as PILImage
                    from torchvision import transforms as T

                    seg_pre_path = data_dict.get("seg_pre")
                    seg_post_path = data_dict.get("seg_post")

                    seg_transform = T.Compose([T.Resize(1024, interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(1024), T.ToTensor()])
                    seg_src_cond = seg_transform(PILImage.open(seg_pre_path).convert("RGB")).unsqueeze(0).to(device).half()
                    seg_tar_cond = seg_transform(PILImage.open(seg_post_path).convert("RGB")).unsqueeze(0).to(device).half()

                    controlnet_conditioning_scale = exp_dict.get("controlnet_conditioning_scale", 1.0)

                    x0_tar = FlowEditSD3ControlNet(
                        pipe, scheduler, controlnet, x0_src,
                        src_prompt, tar_prompt, negative_prompt,
                        seg_src_cond, seg_tar_cond,
                        T_steps, n_avg,
                        src_guidance_scale, tar_guidance_scale,
                        n_min, n_max,
                        controlnet_conditioning_scale,
                    )
```

- [ ] **Step 3: Move controlnet to device after pipe**

After `pipe = pipe.to(device)` (line 43), add:

```python
    if model_type == 'SD3_ControlNet':
        controlnet = controlnet.to(device)
        controlnet.eval()
```

- [ ] **Step 4: Verify parse**

Run: `cd /home/summertide/Workspace/FlowEdit_dev && python -c "import ast; ast.parse(open('run_script.py').read()); print('Syntax OK')"`

Expected: `Syntax OK`

- [ ] **Step 5: Commit**

```bash
git add run_script.py
git commit -m "feat: add SD3_ControlNet model type to run_script.py"
```

---

## Task 11: End-to-End Smoke Test

**Files:**
- No new files — verification only

- [ ] **Step 1: Verify all modules import correctly**

Run:
```bash
cd /home/summertide/Workspace/FlowEdit_dev && python -c "
from rs_data.class_mapping import segmap_to_rgb, segmap_to_text, HIUCD_CLASSES
from rs_data.rs_dataset import RSControlNetDataset
from FlowEdit_utils import FlowEditSD3ControlNet, calc_v_sd3_controlnet
import ast
for f in ['train_controlnet_sd3_rs.py', 'rs_inference.py', 'rs_evaluate.py', 'run_script.py']:
    ast.parse(open(f).read())
print('All modules OK')
"
```

Expected: `All modules OK`

- [ ] **Step 2: Run a dry test of class mapping with synthetic data**

Run:
```bash
cd /home/summertide/Workspace/FlowEdit_dev && python -c "
import numpy as np
from rs_data.class_mapping import segmap_to_rgb, segmap_to_text

# Simulate a segmap: mostly buildings (5) and roads (7)
seg = np.zeros((64, 64), dtype=np.int32)
seg[:40, :] = 5  # buildings
seg[40:, :30] = 7  # roads
seg[40:, 30:] = 3  # low vegetation

rgb = segmap_to_rgb(seg)
text = segmap_to_text(seg)
print(f'RGB shape: {rgb.shape}')
print(f'Text: {text}')
print(f'Buildings color: {rgb[0, 0].tolist()}')
print(f'Roads color: {rgb[50, 0].tolist()}')
assert rgb[0, 0].tolist() == [255, 0, 0], 'Building color wrong'
assert rgb[50, 0].tolist() == [255, 128, 0], 'Road color wrong'
assert 'buildings' in text
print('PASS')
"
```

Expected: `PASS`

- [ ] **Step 3: Commit any fixes if needed, then tag milestone**

```bash
git add -A
git commit -m "milestone: RS FlowEdit codebase complete, ready for data prep and training"
```

---

## Execution Order Summary

```
Task 1:  Class mapping utilities          (no dependencies)
Task 2:  Data preparation script          (depends on Task 1)
Task 3:  PyTorch Dataset                  (depends on Task 1)
Task 4:  ControlNet training script       (depends on Task 3)
Task 5:  calc_v_sd3_controlnet()          (no dependencies)
Task 6:  FlowEditSD3ControlNet()          (depends on Task 5)
Task 7:  Experiment configs               (no dependencies)
Task 8:  Batch inference script            (depends on Tasks 1, 5, 6)
Task 9:  Evaluation script                (no dependencies)
Task 10: Update run_script.py             (depends on Tasks 5, 6)
Task 11: End-to-end smoke test            (depends on all above)
```

Parallelizable groups:
- **Group A** (Tasks 1, 5, 7, 9): independent, can run in parallel
- **Group B** (Tasks 2, 3, 6): depend on Group A
- **Group C** (Tasks 4, 8, 10): depend on Group B
- **Group D** (Task 11): final verification
