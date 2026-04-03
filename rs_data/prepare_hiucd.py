"""Prepare Hi-UCD dataset for ControlNet training.

Splits bi-temporal pairs into single-temporal (image, segmap_rgb, text) samples.
Outputs a training manifest CSV.

Actual Hi-UCD directory structure:
    hiucd_root/
    ├── train/
    │   ├── image/
    │   │   ├── 2018/        # pre-temporal images (512x512)
    │   │   └── 2019/        # post-temporal images (512x512)
    │   └── mask/
    │       └── 2018_2019/   # change masks (R=pre_class, G=post_class, B=unchanged_flag)
    ├── val/
    │   ├── image/2018/, 2019/
    │   └── mask/2018_2019/
    └── test/
        └── image/2018/, 2019/   # no masks

Usage:
    python -m rs_data.prepare_hiucd --hiucd_root /path/to/Hi-UCD --output_dir ./data/hiucd_prepared
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

from rs_data.hiucd import parse_hiucd_mask, hiucd_segmap_to_rgb, hiucd_segmap_to_text


def prepare_split(hiucd_root: str, split: str, output_dir: str) -> list:
    """Process one split (train/val) of Hi-UCD.

    For each image pair + change mask, produces two single-temporal samples:
    - (pre_image, pre_segmap_rgb, pre_text)
    - (post_image, post_segmap_rgb, post_text)

    Returns list of dicts: {image_path, segmap_path, text_prompt, phase, original_name}
    """
    split_dir = Path(hiucd_root) / split
    pre_img_dir = split_dir / "image" / "2018"
    post_img_dir = split_dir / "image" / "2019"
    mask_dir = split_dir / "mask" / "2018_2019"

    if not mask_dir.exists():
        print(f"Warning: {mask_dir} not found, skipping split '{split}'")
        return []

    out_dir = Path(output_dir) / split
    out_img_dir = out_dir / "images"
    out_seg_dir = out_dir / "segmaps"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    records = []
    mask_files = sorted(mask_dir.glob("*.png"))

    for mask_path in mask_files:
        stem = mask_path.stem
        pre_img_path = pre_img_dir / f"{stem}.png"
        post_img_path = post_img_dir / f"{stem}.png"

        if not pre_img_path.exists() or not post_img_path.exists():
            print(f"Warning: missing image for {stem}, skipping")
            continue

        # Load change mask and parse into pre/post segmentation maps
        mask_rgb = np.array(Image.open(mask_path))
        seg_pre, seg_post, _change_label = parse_hiucd_mask(mask_rgb)

        # Process both temporal phases
        for phase, img_path, seg_map in [("pre", pre_img_path, seg_pre), ("post", post_img_path, seg_post)]:
            image = Image.open(img_path).convert("RGB")

            # Convert class indices to RGB visualization
            seg_rgb = hiucd_segmap_to_rgb(seg_map)
            text_prompt = hiucd_segmap_to_text(seg_map)

            # Save processed files
            out_name = f"{phase}_{stem}"
            out_img_path = out_img_dir / f"{out_name}.png"
            out_seg_path = out_seg_dir / f"{out_name}.png"

            image.save(out_img_path)
            Image.fromarray(seg_rgb).save(out_seg_path)

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
    for split in ["train", "val"]:
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
