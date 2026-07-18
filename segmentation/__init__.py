from .classical import canny_segmentation, otsu_segmentation, watershed_segmentation, segment_classical
from .deep_learning import UNet

__all__ = [
    "canny_segmentation",
    "otsu_segmentation",
    "watershed_segmentation",
    "segment_classical",
    "UNet",
]
