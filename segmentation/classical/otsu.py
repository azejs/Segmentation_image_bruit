"""
Otsu's thresholding-based segmentation.
Otsu's method automatically finds the optimal global threshold by minimising
intra-class variance (equivalent to maximising inter-class variance).
"""

import cv2
import numpy as np


def otsu_segmentation(
    image: np.ndarray,
    blur_kernel: int = 5,
    apply_blur: bool = True,
    morph_cleanup: bool = True,
) -> np.ndarray:
    """
    Segment an image using Otsu's global thresholding.

    Args:
        image:         Input image (uint8, gray or colour).
        blur_kernel:   Gaussian kernel size applied before thresholding (odd).
        apply_blur:    Whether to apply Gaussian blur before thresholding.
        morph_cleanup: If True, apply morphological opening+closing to remove
                       small artifacts.

    Returns:
        Binary segmentation mask (uint8, 0 or 255).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if apply_blur:
        ksize = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if morph_cleanup:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def adaptive_threshold_segmentation(
    image: np.ndarray,
    block_size: int = 11,
    C: int = 2,
) -> np.ndarray:
    """
    Adaptive (local) thresholding – handles uneven illumination.

    Args:
        image:      Input image (uint8, gray or colour).
        block_size: Size of the local neighbourhood (odd integer).
        C:          Constant subtracted from the local mean.

    Returns:
        Binary segmentation mask (uint8, 0 or 255).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    bsize = block_size if block_size % 2 == 1 else block_size + 1
    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bsize, C,
    )
    return mask
