"""
Canny edge detection-based segmentation.
The Canny filter detects contours using gradient magnitude + non-maximum
suppression + hysteresis thresholding.
"""

import cv2
import numpy as np


def canny_segmentation(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_kernel: int = 5,
    apply_blur: bool = True,
) -> np.ndarray:
    """
    Detect edges with the Canny algorithm.

    Args:
        image:          Input image (uint8, gray or colour).
        low_threshold:  Lower bound for hysteresis thresholding.
        high_threshold: Upper bound for hysteresis thresholding.
        blur_kernel:    Gaussian kernel size applied before Canny (odd integer).
        apply_blur:     Whether to apply Gaussian blur before edge detection.

    Returns:
        Binary edge map (uint8, 0 or 255).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if apply_blur:
        ksize = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def canny_auto_threshold(image: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    """
    Compute Canny thresholds automatically from the image median intensity
    (Otsu-inspired heuristic).

    Args:
        image: Grayscale or colour image.
        sigma: Controls how far the thresholds deviate from the median.

    Returns:
        (low_threshold, high_threshold)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    v = np.median(gray)
    low  = int(max(0,   (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    return low, high
