from pathlib import Path

ROOT = Path(__file__).parent

DATA_DIR       = ROOT / "data"
IMAGE_DIR      = DATA_DIR / "images"
MASK_DIR       = DATA_DIR / "masks"
RESULTS_DIR    = ROOT / "results"
FIGURES_DIR    = RESULTS_DIR / "figures"
METRICS_DIR    = RESULTS_DIR / "metrics"

# Noise levels tested (standard deviation for Gaussian, density for S&P)
GAUSSIAN_SIGMAS  = [10, 25, 50, 75]
SP_DENSITIES     = [0.02, 0.05, 0.10, 0.20]

# Canny thresholds
CANNY_LOW  = 50
CANNY_HIGH = 150

# U-Net training
UNET_IMAGE_SIZE  = 256
UNET_BATCH_SIZE  = 8
UNET_EPOCHS      = 50
UNET_LR          = 1e-4
UNET_CHECKPOINT  = ROOT / "segmentation" / "deep_learning" / "unet_best.pth"

# Evaluation
METRICS = ["iou", "dice", "precision", "recall", "f1"]
