"""
Training loop for U-Net.
Supports mixed noise augmentation to improve robustness.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from .unet import UNet
from .dataset import SegmentationDataset


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation (complement of Dice coefficient)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num   = 2.0 * (probs * targets).sum() + self.smooth
        den   = probs.sum() + targets.sum() + self.smooth
        return 1.0 - num / den


class CombinedLoss(nn.Module):
    """BCE + Dice combined loss (best for segmentation)."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


def train_unet(
    image_dir: str | Path,
    mask_dir: str | Path,
    checkpoint_path: str | Path,
    image_size: int = 256,
    batch_size: int = 8,
    epochs: int = 50,
    lr: float = 1e-4,
    val_split: float = 0.15,
    noise_type: str | None = "gaussian",
    noise_params: dict | None = None,
    device: str | None = None,
) -> dict:
    """
    Train a U-Net model and save the best checkpoint (by validation loss).

    Args:
        image_dir:       Directory of training images.
        mask_dir:        Directory of binary masks.
        checkpoint_path: Where to save the best model weights (.pth).
        image_size:      Spatial resolution (square).
        batch_size:      Training batch size.
        epochs:          Number of training epochs.
        lr:              Adam learning rate.
        val_split:       Fraction of data used for validation.
        noise_type:      Noise type for augmentation ('gaussian', 'salt_and_pepper', None).
        noise_params:    Parameters for the noise function.
        device:          'cuda', 'cpu', or None (auto-detect).

    Returns:
        Dictionary with training history (train_loss, val_loss per epoch).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if noise_params is None:
        noise_params = {"sigma": 25}

    # Dataset
    dataset = SegmentationDataset(
        image_dir, mask_dir,
        image_size=image_size,
        noise_type=noise_type,
        noise_params=noise_params,
        augment=True,
    )

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    # Disable augmentation / noise on validation split
    val_ds.dataset.augment    = False
    val_ds.dataset.noise_type = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model, loss, optimiser, scheduler
    model     = UNet(in_channels=3, out_channels=1).to(device)
    criterion = CombinedLoss(bce_weight=0.5)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5, verbose=True)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history    = {"train_loss": [], "val_loss": []}
    best_loss  = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)

            optimiser.zero_grad()
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]  ", leave=False):
                images = batch["image"].to(device)
                masks  = batch["mask"].to(device)
                logits = model(images)
                loss   = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Best model saved to {checkpoint_path}")

    return history
