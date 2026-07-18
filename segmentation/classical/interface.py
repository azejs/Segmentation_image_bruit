"""
Unified interface for all classical segmentation methods.
"""

import numpy as np
from .canny import canny_segmentation
from .otsu import otsu_segmentation, adaptive_threshold_segmentation
from .watershed import watershed_segmentation


METHOD_MAP = {
    "canny":     canny_segmentation,
    "otsu":      otsu_segmentation,
    "adaptive":  adaptive_threshold_segmentation,
    "watershed": lambda img, **kw: watershed_segmentation(img, **kw)[0],
}


def segment_classical(image: np.ndarray, method: str = "otsu", **kwargs) -> np.ndarray:
    """
    Segment an image using a classical method.

    Args:
        image:  Input image (uint8).
        method: One of 'canny', 'otsu', 'adaptive', 'watershed'.
        **kwargs: Forwarded to the specific segmentation function.

    Returns:
        Segmentation result (binary mask or label map depending on method).
    """
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHOD_MAP.keys())}")
    return METHOD_MAP[method](image, **kwargs)
