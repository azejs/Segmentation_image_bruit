"""
Classical denoising filters used as pre-processing before segmentation.

Filters implemented:
  - Gaussian blur
  - Median filter
  - Bilateral filter        (edge-preserving)
  - Non-Local Means (NLM)   (patch-based, best quality)
"""

import cv2
import numpy as np


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """
    Gaussian blur – effective against Gaussian noise.

    Args:
        image:       Input image (uint8).
        kernel_size: Must be odd. Larger → more smoothing.
        sigma:       Gaussian std; 0 lets OpenCV choose from kernel_size.

    Returns:
        Blurred image (uint8).
    """
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Median filter – very effective against Salt-and-Pepper noise.

    Args:
        image:       Input image (uint8).
        kernel_size: Must be odd. Larger → more smoothing.

    Returns:
        Filtered image (uint8).
    """
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(image, ksize)


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Bilateral filter – smooths while preserving edges.
    Good compromise for mixed noise.

    Args:
        image:       Input image (uint8).
        d:           Diameter of each pixel neighbourhood.
        sigma_color: Filter sigma in colour space.
        sigma_space: Filter sigma in coordinate space.

    Returns:
        Filtered image (uint8).
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def nlm_filter(
    image: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """
    Non-Local Means denoising – patch-based, best quality but slowest.

    Args:
        image:           Input image (uint8, gray or colour).
        h:               Filtering strength (higher → more smoothing, less detail).
        template_window: Size of the patch used to compute weights (odd).
        search_window:   Size of the window to search for similar patches (odd).

    Returns:
        Denoised image (uint8).
    """
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
    else:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window, search_window)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

FILTER_MAP = {
    "gaussian":  gaussian_filter,
    "median":    median_filter,
    "bilateral": bilateral_filter,
    "nlm":       nlm_filter,
}


def denoise(image: np.ndarray, method: str = "median", **kwargs) -> np.ndarray:
    """
    Unified denoising interface.

    Args:
        image:  Input image (uint8).
        method: One of 'gaussian', 'median', 'bilateral', 'nlm'.
        **kwargs: Forwarded to the specific filter function.

    Returns:
        Denoised image (uint8).
    """
    if method not in FILTER_MAP:
        raise ValueError(f"Unknown filter '{method}'. Choose from: {list(FILTER_MAP.keys())}")
    return FILTER_MAP[method](image, **kwargs)
