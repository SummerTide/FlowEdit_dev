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
            transforms.ToTensor(),  # [0, 1] range — fed directly to ControlNet, not VAE
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
