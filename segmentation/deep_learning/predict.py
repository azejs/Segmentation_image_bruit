"""
Inference helpers for the trained U-Net model.
"""

import cv2
import numpy as np
import torch
from pathlib import Path

from .unet import UNet


def load_model(
    checkpoint_path: str | Path,
    in_channels: int = 3,
    out_channels: int = 1,
    device: str | None = None,
) -> UNet:
    """Load a U-Net from a .pth checkpoint file."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(
    model: UNet,
    image: np.ndarray,
    image_size: int = 256,
    threshold: float = 0.5,
    device: str | None = None,
) -> np.ndarray:
    """
    Run inference on a single image.

    Args:
        model:      Loaded UNet instance (in eval mode).
        image:      Input image (uint8, HxW or HxWx3).
        image_size: Size the image is resized to before inference.
        threshold:  Sigmoid threshold to binarise the output.
        device:     'cuda' or 'cpu' (auto-detected if None).

    Returns:
        Binary mask (uint8, 0/255) resized back to the original image size.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    orig_h, orig_w = image.shape[:2]

    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image_rgb, (image_size, image_size))
    tensor  = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor  = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary = (prob > threshold).astype(np.uint8) * 255
    binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return binary


def predict_batch(
    model: UNet,
    images: list[np.ndarray],
    image_size: int = 256,
    threshold: float = 0.5,
    device: str | None = None,
) -> list[np.ndarray]:
    """Run predict() over a list of images."""
    return [predict(model, img, image_size, threshold, device) for img in images]
