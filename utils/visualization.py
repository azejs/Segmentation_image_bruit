"""
Plotting and visualisation helpers for the PFE report figures.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from pathlib import Path


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "serif",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {path}")


def show_noise_comparison(
    image: np.ndarray,
    noise_type: str = "gaussian",
    levels: list = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """
    Show an image corrupted at several noise levels side-by-side.

    Args:
        image:      Clean input image (uint8, BGR or gray).
        noise_type: 'gaussian' or 'salt_and_pepper'.
        levels:     Noise parameter values.
        figsize:    Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    from noise import add_noise, compute_psnr

    if levels is None:
        levels = [10, 25, 50, 75] if noise_type == "gaussian" else [0.02, 0.05, 0.10, 0.20]

    n = len(levels) + 1
    fig, axes = plt.subplots(1, n, figsize=figsize)

    def _show(ax, img, title):
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    _show(axes[0], image, "Original")
    for ax, lvl in zip(axes[1:], levels):
        kw   = {"sigma": lvl} if noise_type == "gaussian" else {"density": lvl}
        noisy = add_noise(image, noise_type, **kw)
        psnr  = compute_psnr(image, noisy)
        label = f"σ={lvl}" if noise_type == "gaussian" else f"d={lvl}"
        _show(ax, noisy, f"{label}\nPSNR={psnr:.1f} dB")

    fig.suptitle(
        f"Impact du bruit {'Gaussien' if noise_type == 'gaussian' else 'Sel-et-Poivre'}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def show_segmentation_results(
    image: np.ndarray,
    results: dict[str, np.ndarray],
    gt_mask: np.ndarray | None = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """
    Show segmentation results from multiple methods side-by-side.

    Args:
        image:   Input image (BGR or gray).
        results: Dict mapping method name → binary mask.
        gt_mask: Optional ground-truth mask for comparison.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    n_cols = 1 + len(results) + (1 if gt_mask is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    def _show(ax, img, title, cmap="gray"):
        if img is not None and img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    ax_idx = 0
    _show(axes[ax_idx], image, "Image d'entrée")
    ax_idx += 1

    for name, mask in results.items():
        _show(axes[ax_idx], mask, name)
        ax_idx += 1

    if gt_mask is not None:
        _show(axes[ax_idx], gt_mask, "Vérité terrain")

    fig.tight_layout()
    return fig


def plot_metrics_vs_noise(
    df: pd.DataFrame,
    metric: str = "iou",
    noise_type: str = "gaussian",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot a metric vs noise level for several pipelines.

    Args:
        df:         DataFrame from benchmark_noise_levels().
        metric:     Column name: 'iou', 'dice', 'f1', 'precision', 'recall'.
        noise_type: 'gaussian' or 'salt_and_pepper' (for axis label).
        figsize:    Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    subset = df[df["noise_type"] == noise_type].copy()
    agg    = subset.groupby(["pipeline", "noise_level"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    for pipeline, grp in agg.groupby("pipeline"):
        ax.plot(grp["noise_level"], grp[metric], marker="o", label=pipeline)

    xlabel = "Écart-type σ (bruit Gaussien)" if noise_type == "gaussian" \
             else "Densité d (bruit Sel-et-Poivre)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} en fonction du niveau de bruit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_all_metrics(
    df: pd.DataFrame,
    noise_type: str = "gaussian",
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    4-panel figure: IoU, Dice, Precision, Recall vs noise level.
    """
    metrics = ["iou", "dice", "precision", "recall"]
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    subset = df[df["noise_type"] == noise_type].copy()
    xlabel = "σ (Gaussien)" if noise_type == "gaussian" else "densité (S&P)"

    for ax, metric in zip(axes.flat, metrics):
        agg = subset.groupby(["pipeline", "noise_level"])[metric].mean().reset_index()
        for pipeline, grp in agg.groupby("pipeline"):
            ax.plot(grp["noise_level"], grp[metric], marker="o", label=pipeline)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Métriques de segmentation en fonction du bruit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
