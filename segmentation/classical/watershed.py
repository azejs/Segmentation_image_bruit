"""
Watershed-based segmentation.
The watershed algorithm treats the gradient magnitude as a topographic surface
and finds catchment basins (segments) by simulating flooding from markers.
"""

import cv2
import numpy as np


def watershed_segmentation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment an image using the Watershed algorithm.

    Pipeline:
        1. Convert to grayscale and apply Otsu thresholding.
        2. Morphological operations to identify sure foreground/background.
        3. Compute distance transform for foreground markers.
        4. Run cv2.watershed with those markers.

    Args:
        image: Input colour image (uint8, BGR or RGB, HxWx3).

    Returns:
        (label_map, overlay):
            label_map – integer label map (int32), -1 = boundaries.
            overlay   – original image with watershed boundaries drawn in red.
    """
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image.copy()
    else:
        bgr = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Step 1: Otsu thresholding to get binary foreground ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Step 2: Morphological cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground via distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # --- Step 3: Unknown region and markers ---
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # background = 1
    markers[unknown == 255] = 0    # unknown region = 0

    # --- Step 4: Apply Watershed ---
    markers = cv2.watershed(bgr, markers)

    # --- Overlay boundaries on the original image ---
    overlay = bgr.copy()
    overlay[markers == -1] = [0, 0, 255]   # red boundaries

    return markers, overlay
