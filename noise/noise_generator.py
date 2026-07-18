"""
Noise generation functions for image corruption experiments.
Supports Gaussian noise and Salt-and-Pepper noise.
"""

import numpy as np
import cv2


def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Add Gaussian (additive white) noise to an image.

    Args:
        image: Input image (uint8, HxW or HxWxC).
        sigma: Standard deviation of the Gaussian noise (0–255 scale).

    Returns:
        Noisy image clipped to [0, 255] as uint8.
    """
    image_float = image.astype(np.float32)
    noise = np.random.normal(0, sigma, image_float.shape).astype(np.float32)
    noisy = np.clip(image_float + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_and_pepper_noise(image: np.ndarray, density: float = 0.05) -> np.ndarray:
    """
    Add Salt-and-Pepper (impulse) noise to an image.

    Args:
        image: Input image (uint8, HxW or HxWxC).
        density: Total proportion of pixels to corrupt (split equally between
                 salt and pepper).

    Returns:
        Noisy image as uint8.
    """
    noisy = image.copy()
    h, w = image.shape[:2]
    n_pixels = h * w

    # Salt (white pixels)
    n_salt = int(n_pixels * density / 2)
    salt_coords = (
        np.random.randint(0, h, n_salt),
        np.random.randint(0, w, n_salt),
    )
    if image.ndim == 3:
        noisy[salt_coords[0], salt_coords[1]] = 255
    else:
        noisy[salt_coords] = 255

    # Pepper (black pixels)
    n_pepper = int(n_pixels * density / 2)
    pepper_coords = (
        np.random.randint(0, h, n_pepper),
        np.random.randint(0, w, n_pepper),
    )
    if image.ndim == 3:
        noisy[pepper_coords[0], pepper_coords[1]] = 0
    else:
        noisy[pepper_coords] = 0

    return noisy


def add_noise(image: np.ndarray, noise_type: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Unified noise addition interface.

    Args:
        image:      Input image.
        noise_type: "gaussian" or "salt_and_pepper".
        **kwargs:   Parameters forwarded to the specific noise function.

    Returns:
        Noisy image.
    """
    if noise_type == "gaussian":
        return add_gaussian_noise(image, **kwargs)
    elif noise_type in ("salt_and_pepper", "sp"):
        return add_salt_and_pepper_noise(image, **kwargs)
    else:
        raise ValueError(f"Unknown noise type: '{noise_type}'. Use 'gaussian' or 'salt_and_pepper'.")


def compute_psnr(original: np.ndarray, noisy: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two images."""
    mse = np.mean((original.astype(np.float64) - noisy.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_snr(original: np.ndarray, noisy: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio (dB)."""
    signal_power = np.mean(original.astype(np.float64) ** 2)
    noise_power  = np.mean((original.astype(np.float64) - noisy.astype(np.float64)) ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)
