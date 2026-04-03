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
