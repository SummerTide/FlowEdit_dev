"""Build a small Hi-UCD subset from the validation split for local testing.

The subset keeps the original Hi-UCD layout so it can be consumed directly by
`rs_inference.py`:

    output_root/
    └── mini_val/
        ├── image/
        │   ├── 2018/
        │   └── 2019/
        └── mask/
            └── 2018_2019/

Usage:
    python -m rs_data.build_hiucd_small_testset \
        --val_dir /home/summertide/Workspace/Hi-UCD/val \
        --output_root ./data/hiucd_small_test \
        --output_split mini_val \
        --num_pairs 50
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from rs_data.hiucd import HIUCD_CHANGE_CHANGED, HIUCD_CHANGE_UNLABELED, parse_hiucd_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Build a small Hi-UCD subset from val")
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to the original Hi-UCD val directory",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/hiucd_small_test",
        help="Output dataset root",
    )
    parser.add_argument(
        "--output_split",
        type=str,
        default="mini_val",
        help="Name of the split directory created under output_root",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=50,
        help="Number of image/mask pairs to copy",
    )
    return parser.parse_args()


def collect_ranked_pairs(val_dir: Path):
    pre_dir = val_dir / "image" / "2018"
    post_dir = val_dir / "image" / "2019"
    mask_dir = val_dir / "mask" / "2018_2019"

    if not pre_dir.exists() or not post_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            "Expected val_dir to contain image/2018, image/2019, and mask/2018_2019"
        )

    rows = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        stem = mask_path.stem
        pre_img = pre_dir / f"{stem}.png"
        post_img = post_dir / f"{stem}.png"
        if not pre_img.exists() or not post_img.exists():
            continue

        mask_rgb = np.array(Image.open(mask_path))
        _, _, change_label = parse_hiucd_mask(mask_rgb)

        changed_pixels = int((change_label == HIUCD_CHANGE_CHANGED).sum())
        valid_pixels = int((change_label != HIUCD_CHANGE_UNLABELED).sum())
        total_pixels = int(change_label.size)
        if valid_pixels == 0:
            continue

        rows.append(
            {
                "id": stem,
                "changed_pixels": changed_pixels,
                "valid_pixels": valid_pixels,
                "total_pixels": total_pixels,
                "change_ratio_valid": changed_pixels / valid_pixels,
                "change_ratio_total": changed_pixels / total_pixels,
                "pre_image_src": str(pre_img),
                "post_image_src": str(post_img),
                "mask_src": str(mask_path),
            }
        )

    rows.sort(
        key=lambda row: (
            row["change_ratio_total"],
            row["changed_pixels"],
            row["change_ratio_valid"],
        ),
        reverse=True,
    )
    return rows


def copy_subset(rows, output_root: Path, output_split: str):
    split_dir = output_root / output_split
    out_pre_dir = split_dir / "image" / "2018"
    out_post_dir = split_dir / "image" / "2019"
    out_mask_dir = split_dir / "mask" / "2018_2019"

    out_pre_dir.mkdir(parents=True, exist_ok=True)
    out_post_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row in rows:
        stem = row["id"]
        out_pre = out_pre_dir / f"{stem}.png"
        out_post = out_post_dir / f"{stem}.png"
        out_mask = out_mask_dir / f"{stem}.png"

        shutil.copy2(row["pre_image_src"], out_pre)
        shutil.copy2(row["post_image_src"], out_post)
        shutil.copy2(row["mask_src"], out_mask)

        manifest_rows.append(
            {
                **row,
                "pre_image_out": str(out_pre),
                "post_image_out": str(out_post),
                "mask_out": str(out_mask),
            }
        )

    manifest_path = split_dir / "selected_pairs.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    ids_path = split_dir / "selected_ids.txt"
    ids_path.write_text("\n".join(row["id"] for row in manifest_rows) + "\n")

    return manifest_path, ids_path


def main():
    args = parse_args()

    val_dir = Path(args.val_dir)
    output_root = Path(args.output_root)
    ranked_rows = collect_ranked_pairs(val_dir)
    if not ranked_rows:
        raise RuntimeError(f"No valid pairs found under {val_dir}")

    selected_rows = ranked_rows[: args.num_pairs]
    manifest_path, ids_path = copy_subset(selected_rows, output_root, args.output_split)

    print(
        f"Built {len(selected_rows)} pairs at {output_root / args.output_split}. "
        f"Manifest: {manifest_path}. IDs: {ids_path}"
    )


if __name__ == "__main__":
    main()
