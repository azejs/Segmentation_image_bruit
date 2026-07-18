from .canny import canny_segmentation
from .otsu import otsu_segmentation
from .watershed import watershed_segmentation
from .interface import segment_classical

__all__ = [
    "canny_segmentation",
    "otsu_segmentation",
    "watershed_segmentation",
    "segment_classical",
]
