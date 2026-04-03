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
