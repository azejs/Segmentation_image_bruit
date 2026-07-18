"""
Benchmark utilities: evaluate segmentation pipelines across noise levels
and produce summary DataFrames ready for plotting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable

from .metrics import evaluate_mask
from noise import add_noise, compute_psnr
from denoising import denoise


def benchmark_pipeline(
    images: list[np.ndarray],
    masks_gt: list[np.ndarray],
    segmentor: Callable[[np.ndarray], np.ndarray],
    noise_type: str = "gaussian",
    noise_levels: list = None,
    denoising_method: str | None = "median",
    denoising_params: dict | None = None,
    label: str = "pipeline",
) -> pd.DataFrame:
    """
    Evaluate a segmentation pipeline (optional denoiser + segmentor) across
    multiple noise levels.

    Args:
        images:           List of clean test images.
        masks_gt:         Corresponding ground-truth binary masks.
        segmentor:        Callable(image) -> binary mask.
        noise_type:       'gaussian' or 'salt_and_pepper'.
        noise_levels:     List of noise parameter values (sigma for Gaussian,
                          density for S&P).
        denoising_method: Pre-processing filter name or None to skip.
        denoising_params: Extra kwargs for the denoising function.
        label:            Name tag for this pipeline configuration.

    Returns:
        DataFrame with columns: pipeline, noise_type, noise_level, psnr,
        iou, dice, precision, recall, f1  (one row per image × noise level).
    """
    if noise_levels is None:
        if noise_type == "gaussian":
            noise_levels = [0, 10, 25, 50, 75]
        else:
            noise_levels = [0, 0.02, 0.05, 0.10, 0.20]

    if denoising_params is None:
        denoising_params = {}

    rows = []
    for level in noise_levels:
        for img, gt in zip(images, masks_gt):
            # Add noise
            if noise_type == "gaussian":
                noisy = add_noise(img, "gaussian", sigma=float(level)) if level > 0 else img.copy()
            else:
                noisy = add_noise(img, "salt_and_pepper", density=float(level)) if level > 0 else img.copy()

            psnr = compute_psnr(img, noisy)

            # Optional denoising
            processed = denoise(noisy, denoising_method, **denoising_params) if denoising_method else noisy

            # Segmentation
            pred = segmentor(processed)

            # Metrics
            m = evaluate_mask(pred, gt)
            rows.append({
                "pipeline":    label,
                "noise_type":  noise_type,
                "noise_level": level,
                "psnr_db":     psnr,
                **m,
            })

    return pd.DataFrame(rows)


def benchmark_noise_levels(
    images: list[np.ndarray],
    masks_gt: list[np.ndarray],
    pipelines: dict[str, dict],
    noise_type: str = "gaussian",
    noise_levels: list = None,
) -> pd.DataFrame:
    """
    Run benchmark_pipeline for multiple named pipelines and concatenate results.

    Args:
        images:       List of clean test images.
        masks_gt:     Corresponding ground-truth binary masks.
        pipelines:    Dict mapping pipeline label → dict with keys:
                        'segmentor', optionally 'denoising_method', 'denoising_params'.
        noise_type:   'gaussian' or 'salt_and_pepper'.
        noise_levels: Noise parameter values to test.

    Returns:
        Concatenated DataFrame from all pipelines.
    """
    all_results = []
    for label, cfg in pipelines.items():
        df = benchmark_pipeline(
            images, masks_gt,
            segmentor=cfg["segmentor"],
            noise_type=noise_type,
            noise_levels=noise_levels,
            denoising_method=cfg.get("denoising_method"),
            denoising_params=cfg.get("denoising_params", {}),
            label=label,
        )
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)
