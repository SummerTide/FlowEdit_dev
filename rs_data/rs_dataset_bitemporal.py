"""PyTorch Dataset for bi-temporal ControlNet baseline training on Hi-UCD.

Loads matched bi-temporal pairs directly from Hi-UCD directory structure:
    (pre_image, pre_segmap_rgb, post_segmap_rgb, post_image, text_prompt)

The baseline ControlNet is trained to generate post-images conditioned on
(pre_image, pre_segmap, post_segmap), starting from pure noise.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from rs_data.hiucd import parse_hiucd_mask, hiucd_segmap_to_rgb, hiucd_segmap_to_text


class HiUCDBiTemporalDataset(Dataset):
    """Loads bi-temporal pairs from Hi-UCD for conditional generation training.

    Each sample yields:
        - pre_image: pre-temporal RGB image (target for VAE encoding as condition)
        - post_image: post-temporal RGB image (generation target)
        - pre_segmap: pre-temporal segmap RGB (condition)
        - post_segmap: post-temporal segmap RGB (condition)
        - caption: text prompt derived from post-temporal segmap
    """

    def __init__(self, hiucd_root: str, split: str = "train", resolution: int = 512):
        self.resolution = resolution
        self.augment = (split == "train")

        split_dir = Path(hiucd_root) / split
        self.pre_img_dir = split_dir / "image" / "2018"
        self.post_img_dir = split_dir / "image" / "2019"
        self.mask_dir = split_dir / "mask" / "2018_2019"

        # Collect valid pairs (mask + both images exist)
        self.stems = []
        self.prompts = []
        for mask_path in sorted(self.mask_dir.glob("*.png")):
            stem = mask_path.stem
            if (self.pre_img_dir / f"{stem}.png").exists() and \
               (self.post_img_dir / f"{stem}.png").exists():
                # Pre-parse mask to extract text prompt (avoids re-parsing in __getitem__)
                mask_rgb = np.array(Image.open(mask_path))
                _, seg_post_np, _ = parse_hiucd_mask(mask_rgb)
                self.stems.append(stem)
                self.prompts.append(hiucd_segmap_to_text(seg_post_np))

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
            transforms.Normalize([0.5], [0.5]),  # scale to [-1, 1] for VAE encoding
        ])

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        pre_img = Image.open(self.pre_img_dir / f"{stem}.png").convert("RGB")
        post_img = Image.open(self.post_img_dir / f"{stem}.png").convert("RGB")

        mask_rgb = np.array(Image.open(self.mask_dir / f"{stem}.png"))
        seg_pre_np, seg_post_np, _ = parse_hiucd_mask(mask_rgb)
        seg_pre_pil = Image.fromarray(hiucd_segmap_to_rgb(seg_pre_np))
        seg_post_pil = Image.fromarray(hiucd_segmap_to_rgb(seg_post_np))

        # Consistent augmentation across all four images
        if self.augment:
            if random.random() > 0.5:
                pre_img = pre_img.transpose(Image.FLIP_LEFT_RIGHT)
                post_img = post_img.transpose(Image.FLIP_LEFT_RIGHT)
                seg_pre_pil = seg_pre_pil.transpose(Image.FLIP_LEFT_RIGHT)
                seg_post_pil = seg_post_pil.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                pre_img = pre_img.transpose(Image.FLIP_TOP_BOTTOM)
                post_img = post_img.transpose(Image.FLIP_TOP_BOTTOM)
                seg_pre_pil = seg_pre_pil.transpose(Image.FLIP_TOP_BOTTOM)
                seg_post_pil = seg_post_pil.transpose(Image.FLIP_TOP_BOTTOM)
            rot = random.choice([0, 90, 180, 270])
            if rot > 0:
                pre_img = pre_img.rotate(rot, expand=False)
                post_img = post_img.rotate(rot, expand=False)
                seg_pre_pil = seg_pre_pil.rotate(rot, expand=False)
                seg_post_pil = seg_post_pil.rotate(rot, expand=False)

        return {
            "pre_image": self.image_transform(pre_img),
            "post_image": self.image_transform(post_img),
            "pre_segmap": self.seg_transform(seg_pre_pil),
            "post_segmap": self.seg_transform(seg_post_pil),
            "caption": self.prompts[idx],
        }
