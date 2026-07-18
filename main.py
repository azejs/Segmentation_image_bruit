"""
main.py – Point d'entrée principal du projet PFE.

Usage:
    python main.py --mode demo
    python main.py --mode benchmark --noise gaussian
    python main.py --mode train
    python main.py --mode predict --image path/to/image.png
"""

import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

import config
from noise import add_noise, compute_psnr
from denoising import denoise
from segmentation.classical import segment_classical
from evaluation import benchmark_noise_levels
from utils import (
    load_image, load_mask, load_dataset, save_image,
    show_noise_comparison, show_segmentation_results,
    plot_metrics_vs_noise, plot_all_metrics, save_figure,
)


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode: visualise noise types + segmentation methods on one image
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(image_path: Path | None = None):
    """Quick visual demonstration on a single image."""
    if image_path is None:
        # Fall back to first image in data/images/
        candidates = list(config.IMAGE_DIR.glob("*.png")) + list(config.IMAGE_DIR.glob("*.jpg"))
        if not candidates:
            print("[Demo] No images found in data/images/. Generating a synthetic test image.")
            image = _synthetic_image()
        else:
            image = load_image(candidates[0])
    else:
        image = load_image(image_path)

    print(f"[Demo] Image shape: {image.shape}")

    # 1. Noise visualisation
    for noise_type in ("gaussian", "salt_and_pepper"):
        fig = show_noise_comparison(image, noise_type=noise_type)
        save_figure(fig, config.FIGURES_DIR / f"noise_{noise_type}.png")

    # 2. Segmentation comparison (on a Gaussian-noisy image, σ=25)
    noisy = add_noise(image, "gaussian", sigma=25)

    results = {}
    for method in ("canny", "otsu", "adaptive"):
        mask = segment_classical(noisy, method)
        results[method] = mask
        # Also test with median pre-filtering
        filtered = denoise(noisy, "median")
        results[f"median+{method}"] = segment_classical(filtered, method)

    fig = show_segmentation_results(noisy, results)
    save_figure(fig, config.FIGURES_DIR / "segmentation_comparison.png")
    print("[Demo] Figures saved to results/figures/")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark mode: evaluate all pipelines across noise levels
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(noise_type: str = "gaussian"):
    """Run benchmark across all pipelines and save CSV + figures."""
    images, masks_gt, names = load_dataset(config.IMAGE_DIR, config.MASK_DIR)

    if not images:
        print("[Benchmark] No paired (image, mask) data found. Using synthetic data.")
        images, masks_gt = _synthetic_dataset(n=5)

    # Define pipelines: {label: {segmentor, denoising_method}}
    pipelines = {
        "Canny brut":            {"segmentor": lambda img: segment_classical(img, "canny"),
                                   "denoising_method": None},
        "Canny + Gaussien":      {"segmentor": lambda img: segment_classical(img, "canny"),
                                   "denoising_method": "gaussian"},
        "Canny + Médian":        {"segmentor": lambda img: segment_classical(img, "canny"),
                                   "denoising_method": "median"},
        "Canny + Bilatéral":     {"segmentor": lambda img: segment_classical(img, "canny"),
                                   "denoising_method": "bilateral"},
        "Otsu brut":             {"segmentor": lambda img: segment_classical(img, "otsu"),
                                   "denoising_method": None},
        "Otsu + Médian":         {"segmentor": lambda img: segment_classical(img, "otsu"),
                                   "denoising_method": "median"},
        "Otsu + NLM":            {"segmentor": lambda img: segment_classical(img, "otsu"),
                                   "denoising_method": "nlm"},
    }

    noise_levels = (
        config.GAUSSIAN_SIGMAS  if noise_type == "gaussian"
        else config.SP_DENSITIES
    )

    print(f"[Benchmark] Running {len(pipelines)} pipelines × {len(noise_levels)} noise levels "
          f"× {len(images)} images …")

    df = benchmark_noise_levels(images, masks_gt, pipelines,
                                noise_type=noise_type,
                                noise_levels=[0] + list(noise_levels))

    # Save CSV
    csv_path = config.METRICS_DIR / f"benchmark_{noise_type}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[Benchmark] Results saved → {csv_path}")

    # Figures
    for metric in ("iou", "dice", "f1"):
        fig = plot_metrics_vs_noise(df, metric=metric, noise_type=noise_type)
        save_figure(fig, config.FIGURES_DIR / f"{metric}_vs_{noise_type}.png")

    fig_all = plot_all_metrics(df, noise_type=noise_type)
    save_figure(fig_all, config.FIGURES_DIR / f"all_metrics_{noise_type}.png")

    # Summary table
    summary = df.groupby(["pipeline", "noise_level"])[["iou", "dice", "f1"]].mean().round(4)
    print("\n[Benchmark] Summary (mean over all images):")
    print(summary.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Train mode: train U-Net
# ─────────────────────────────────────────────────────────────────────────────

def run_train():
    """Train the U-Net model."""
    try:
        from segmentation.deep_learning import train_unet
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[Train] Missing dependency: {e}. Install with: pip install torch torchvision")
        return

    if not any(config.IMAGE_DIR.iterdir()):
        print("[Train] data/images/ is empty. Please add training images and masks.")
        return

    history = train_unet(
        image_dir=config.IMAGE_DIR,
        mask_dir=config.MASK_DIR,
        checkpoint_path=config.UNET_CHECKPOINT,
        image_size=config.UNET_IMAGE_SIZE,
        batch_size=config.UNET_BATCH_SIZE,
        epochs=config.UNET_EPOCHS,
        lr=config.UNET_LR,
        noise_type="gaussian",
        noise_params={"sigma": 25},
    )

    # Plot training curves
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"],   label="Val Loss")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Perte (BCE + Dice)")
    ax.set_title("Courbes d'apprentissage U-Net")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, config.FIGURES_DIR / "unet_training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Predict mode: run inference on a single image
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(image_path: Path):
    """Run U-Net inference on a single image."""
    try:
        from segmentation.deep_learning import predict
        from segmentation.deep_learning.predict import load_model
    except ImportError as e:
        print(f"[Predict] Missing dependency: {e}")
        return

    if not config.UNET_CHECKPOINT.exists():
        print(f"[Predict] Checkpoint not found: {config.UNET_CHECKPOINT}. Run --mode train first.")
        return

    image = load_image(image_path)
    model = load_model(config.UNET_CHECKPOINT)
    mask  = predict(model, image, image_size=config.UNET_IMAGE_SIZE)

    out_path = config.RESULTS_DIR / f"{image_path.stem}_unet_mask.png"
    save_image(mask, out_path)
    print(f"[Predict] Mask saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers (when no real data is available)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_image(size: int = 256) -> np.ndarray:
    """Generate a simple synthetic test image (white circle on dark background)."""
    img  = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy, r = size // 2, size // 2, size // 3
    cv2.circle(img, (cx, cy), r, (200, 200, 200), -1)
    return img


def _synthetic_dataset(n: int = 5, size: int = 256):
    """Generate n synthetic (image, mask) pairs."""
    images, masks = [], []
    rng = np.random.default_rng(42)
    for _ in range(n):
        img  = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        cx   = rng.integers(size // 4, 3 * size // 4)
        cy   = rng.integers(size // 4, 3 * size // 4)
        r    = rng.integers(size // 6, size // 3)
        cv2.circle(img,  (cx, cy), r, (200, 200, 200), -1)
        cv2.circle(mask, (cx, cy), r, 255,             -1)
        images.append(img)
        masks.append(mask)
    return images, masks


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PFE – Segmentation d'images en présence de bruit"
    )
    parser.add_argument(
        "--mode", choices=["demo", "benchmark", "train", "predict"],
        default="demo", help="Mode d'exécution"
    )
    parser.add_argument(
        "--noise", choices=["gaussian", "salt_and_pepper"],
        default="gaussian", help="Type de bruit pour le benchmark"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Chemin vers l'image (mode predict)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "demo":
        run_demo(Path(args.image) if args.image else None)

    elif args.mode == "benchmark":
        run_benchmark(noise_type=args.noise)

    elif args.mode == "train":
        run_train()

    elif args.mode == "predict":
        if args.image is None:
            print("[Error] --image is required for predict mode.")
            sys.exit(1)
        run_predict(Path(args.image))
