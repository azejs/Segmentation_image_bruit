"""
Segmentation evaluation metrics.

All functions accept binary masks (0/255 uint8 or 0/1 bool/float).
"""

import numpy as np


def _to_binary(mask: np.ndarray) -> np.ndarray:
    """Normalise mask to boolean (True = foreground)."""
    if mask.dtype == bool:
        return mask
    return mask > 127


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Intersection over Union (Jaccard index).

    IoU = |pred ∩ gt| / |pred ∪ gt|
    """
    pred_b = _to_binary(pred)
    gt_b   = _to_binary(gt)
    intersection = np.logical_and(pred_b, gt_b).sum()
    union        = np.logical_or(pred_b,  gt_b).sum()
    return float(intersection / union) if union > 0 else 1.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice coefficient (F1 score for binary masks).

    Dice = 2 |pred ∩ gt| / (|pred| + |gt|)
    """
    pred_b = _to_binary(pred)
    gt_b   = _to_binary(gt)
    intersection = np.logical_and(pred_b, gt_b).sum()
    denom        = pred_b.sum() + gt_b.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


def compute_precision_recall_f1(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Returns:
        (precision, recall, f1)
    """
    pred_b = _to_binary(pred)
    gt_b   = _to_binary(gt)

    tp = np.logical_and(pred_b,  gt_b).sum()
    fp = np.logical_and(pred_b,  ~gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_mask(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute all segmentation metrics for a single (pred, gt) pair.

    Returns:
        Dictionary with keys: iou, dice, precision, recall, f1.
    """
    iou              = compute_iou(pred, gt)
    dice             = compute_dice(pred, gt)
    precision, recall, f1 = compute_precision_recall_f1(pred, gt)

    return {
        "iou":       iou,
        "dice":      dice,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }
