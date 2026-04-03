"""Generic remote sensing class mapping utilities.

Dataset-specific class definitions and mask parsers live in separate modules
(e.g., rs_data/hiucd.py). This module provides shared functions that work
with any dataset's class mapping dict.
"""

import numpy as np


def segmap_to_rgb(segmap_np: np.ndarray, class_mapping: dict) -> np.ndarray:
    """Convert class-index segmentation map (H, W) to RGB image (H, W, 3).

    Args:
        segmap_np: numpy array of shape (H, W) with integer class indices.
        class_mapping: dict of {class_idx: {"name": str, "color": (R, G, B)}}.

    Returns:
        RGB numpy array of shape (H, W, 3), dtype uint8.
    """
    h, w = segmap_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, info in class_mapping.items():
        mask = segmap_np == class_idx
        rgb[mask] = info["color"]
    return rgb


def segmap_to_text(segmap_np: np.ndarray, class_mapping: dict,
                   unlabeled_idx: int = 0, top_k: int = 3) -> str:
    """Generate a text prompt from a segmentation map based on class area ratios.

    Args:
        segmap_np: numpy array of shape (H, W) with integer class indices.
        class_mapping: dict of {class_idx: {"name": str, "color": (R, G, B)}}.
        unlabeled_idx: class index to skip (unlabeled/background).
        top_k: number of top classes (by area) to include in the prompt.

    Returns:
        Text prompt string, e.g. "an aerial remote sensing image with buildings, roads and trees"
    """
    class_counts = {}
    for class_idx, info in class_mapping.items():
        if class_idx == unlabeled_idx:
            continue
        count = np.sum(segmap_np == class_idx)
        if count > 0:
            class_counts[class_idx] = count

    if not class_counts:
        return "an aerial remote sensing image"

    sorted_classes = sorted(class_counts.keys(), key=lambda k: class_counts[k], reverse=True)[:top_k]
    class_names = [class_mapping[c]["name"] for c in sorted_classes]

    if len(class_names) == 1:
        classes_str = class_names[0]
    else:
        classes_str = ", ".join(class_names[:-1]) + " and " + class_names[-1]

    return f"an aerial remote sensing image with {classes_str}"
