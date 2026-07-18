"""
PyTorch Dataset for image segmentation.
Supports optional noise injection at load time to train a noise-robust model.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

from noise import add_noise


class SegmentationDataset(Dataset):
    """
    Loads (image, mask) pairs from two directories.

    Directory conventions:
        image_dir/  – PNG/JPG images
        mask_dir/   – binary masks with the same filename stem

    Args:
        image_dir:   Path to images directory.
        mask_dir:    Path to binary masks directory.
        image_size:  Resize images to (image_size, image_size).
        noise_type:  If not None, apply noise to images ('gaussian' or 'salt_and_pepper').
        noise_params: Dict of kwargs forwarded to add_noise().
        augment:     Apply random horizontal flip + random rotation.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: int = 256,
        noise_type: str | None = None,
        noise_params: dict | None = None,
        augment: bool = False,
    ):
        self.image_dir   = Path(image_dir)
        self.mask_dir    = Path(mask_dir)
        self.image_size  = image_size
        self.noise_type  = noise_type
        self.noise_params = noise_params or {}
        self.augment     = augment

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.image_files = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in exts
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        img_path  = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name

        # Load
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask  = cv2.resize(mask,  (self.image_size, self.image_size),
                           interpolation=cv2.INTER_NEAREST)

        # Optional noise injection
        if self.noise_type:
            image = add_noise(image, self.noise_type, **self.noise_params)

        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask  = cv2.flip(mask,  1)
        if self.augment and np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            M     = cv2.getRotationMatrix2D(
                (self.image_size // 2, self.image_size // 2), angle, 1.0
            )
            image = cv2.warpAffine(image, M, (self.image_size, self.image_size))
            mask  = cv2.warpAffine(mask,  M, (self.image_size, self.image_size))

        # Normalise and convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        # Binary mask: threshold at 127
        mask_binary  = (mask > 127).astype(np.float32)
        mask_tensor  = torch.from_numpy(mask_binary).unsqueeze(0)

        return {"image": image_tensor, "mask": mask_tensor, "name": img_path.stem}
