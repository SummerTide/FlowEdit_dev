"""Hi-UCD dataset class mapping: index → RGB color and text label."""

import numpy as np
from PIL import Image

# Hi-UCD 10-class mapping (ID 0-9)
# Reference: https://github.com/Daisy-7/Hi-UCD-S
HIUCD_CLASSES = {
    0: {"name": "unlabeled", "color": (255, 255, 255)},
    1: {"name": "water", "color": (0, 153, 255)},
    2: {"name": "grass", "color": (202, 255, 122)},
    3: {"name": "buildings", "color": (230, 0, 0)},
    4: {"name": "green house", "color": (230, 0, 255)},
    5: {"name": "roads", "color": (255, 230, 0)},
    6: {"name": "bridge", "color": (255, 181, 197)},
    7: {"name": "others", "color": (0, 255, 230)},
    8: {"name": "bare land", "color": (175, 122, 255)},
    9: {"name": "woodland", "color": (26, 255, 0)},
}


def parse_change_mask(mask_rgb: np.ndarray) -> tuple:
    """Parse Hi-UCD change mask into pre/post segmentation maps and change label.

    Hi-UCD mask encoding (3-channel PNG):
    - Channel 0 (R): T1 (pre) land cover class index
    - Channel 1 (G): T2 (post) land cover class index
    - Channel 2 (B): change label (0=unlabeled, 1=no-change, 2=change)

    Examples:
    - (0, 0, 0) = unlabeled area
    - (3, 3, 1) = building→building, no change
    - (4, 8, 2) = green house→bare land, changed

    Args:
        mask_rgb: numpy array of shape (H, W, 3), dtype uint8.

    Returns:
        Tuple of (seg_pre, seg_post, change_label):
        - seg_pre: (H, W) int32, T1 land cover class indices
        - seg_post: (H, W) int32, T2 land cover class indices
        - change_label: (H, W) uint8, 0=unlabeled, 1=no-change, 2=change
    """
    seg_pre = mask_rgb[:, :, 0].astype(np.int32)
    seg_post = mask_rgb[:, :, 1].astype(np.int32)
    change_label = mask_rgb[:, :, 2]
    return seg_pre, seg_post, change_label


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
        if class_idx == 0:  # skip "unchanged area"
            continue
        count = np.sum(segmap_np == class_idx)
        if count > 0:
            class_counts[class_idx] = count

    if not class_counts:
        return "an aerial remote sensing image"

    sorted_classes = sorted(class_counts.keys(), key=lambda k: class_counts[k], reverse=True)[:top_k]
    class_names = [HIUCD_CLASSES[c]["name"] for c in sorted_classes]

    if len(class_names) == 1:
        classes_str = class_names[0]
    else:
        classes_str = ", ".join(class_names[:-1]) + " and " + class_names[-1]

    return f"an aerial remote sensing image with {classes_str}"
