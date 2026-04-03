"""Hi-UCD dataset adapter.

Hi-UCD (ultra-High Urban Change Detection) dataset.
- Location: Tallinn, Estonia
- Resolution: 0.1m GSD, 512x512 patches
- Temporal: 2018 (T1) → 2019 (T2)
- 10 land cover classes (ID 0-9)
- 3-channel mask encoding: (T1_class, T2_class, change_label)

Reference: https://github.com/Daisy-7/Hi-UCD-S

Directory structure:
    Hi-UCD/
    ├── train/
    │   ├── image/
    │   │   ├── 2018/            # T1 images (512x512 RGB PNG)
    │   │   └── 2019/            # T2 images (512x512 RGB PNG)
    │   └── mask/
    │       └── 2018_2019/       # 3-channel masks (512x512 PNG)
    ├── val/
    │   ├── image/2018/, 2019/
    │   └── mask/2018_2019/
    └── test/
        └── image/2018/, 2019/   # no masks available
"""

import numpy as np

from rs_data.class_mapping import segmap_to_rgb, segmap_to_text

# ============================================================
# Hi-UCD class definition (10 classes, ID 0-9)
# ============================================================
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

HIUCD_UNLABELED_IDX = 0

# Hi-UCD change label encoding (B channel of mask)
HIUCD_CHANGE_UNLABELED = 0
HIUCD_CHANGE_NOCHANGE = 1
HIUCD_CHANGE_CHANGED = 2


# ============================================================
# Hi-UCD mask parser
# ============================================================
def parse_hiucd_mask(mask_rgb: np.ndarray) -> tuple:
    """Parse Hi-UCD 3-channel change mask.

    Hi-UCD mask encoding (3-channel PNG):
    - Channel 0 (R): T1 land cover class index (0-9)
    - Channel 1 (G): T2 land cover class index (0-9)
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


# ============================================================
# Hi-UCD convenience wrappers
# ============================================================
def hiucd_segmap_to_rgb(segmap_np: np.ndarray) -> np.ndarray:
    """Convert Hi-UCD class-index segmap to RGB. See segmap_to_rgb()."""
    return segmap_to_rgb(segmap_np, HIUCD_CLASSES)


def hiucd_segmap_to_text(segmap_np: np.ndarray, top_k: int = 3) -> str:
    """Generate text prompt from Hi-UCD segmap. See segmap_to_text()."""
    return segmap_to_text(segmap_np, HIUCD_CLASSES,
                          unlabeled_idx=HIUCD_UNLABELED_IDX, top_k=top_k)
