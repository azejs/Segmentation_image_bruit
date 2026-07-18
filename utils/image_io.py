"""
Image I/O helpers.
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(path: str | Path, color: bool = True) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path:  File path.
        color: True → BGR (3-channel), False → grayscale.

    Returns:
        Image as uint8 numpy array.
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def load_mask(path: str | Path) -> np.ndarray:
    """Load a binary mask (grayscale, thresholded at 127)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    return (mask > 127).astype(np.uint8) * 255


def load_dataset(
    image_dir: str | Path,
    mask_dir: str | Path,
    color: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Load all (image, mask) pairs from two directories.

    Returns:
        (images, masks, names)
    """
    image_dir = Path(image_dir)
    mask_dir  = Path(mask_dir)
    exts      = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    images, masks, names = [], [], []
    for ip in image_paths:
        mp = mask_dir / ip.name
        if not mp.exists():
            continue
        images.append(load_image(ip, color=color))
        masks.append(load_mask(mp))
        names.append(ip.stem)

    return images, masks, names


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save a numpy array as an image file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
