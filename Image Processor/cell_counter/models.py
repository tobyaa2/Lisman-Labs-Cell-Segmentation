"""Shared data structures passed between pipeline stages.

Kept in a leaf module (no project imports) so segmentation, classification,
metrics and visualization can all depend on it without import cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

# Stable column order for cells.csv.
CSV_COLUMNS = [
    "id",
    "centroid_x",
    "centroid_y",
    "area_px",
    "equiv_diameter_px",
    "solidity",
    "eccentricity",
    "teal_fraction",
    "relative_blueness",
    "median_hue",
    "median_saturation",
    "median_value",
    "is_stained",
    "stain_confidence",
    "on_border",
    "possible_doublet",
    "is_debris",
    "ambiguous",
    "source",
    "sam3_detect_score",
    "sam3_stain_score",
]


@dataclass
class CellRecord:
    """One detected cell: geometry (from the filtered image) + color/classification
    (from the unfiltered image)."""

    id: int
    centroid_x: float
    centroid_y: float
    area_px: int
    equiv_diameter_px: float
    # shape sanity
    solidity: float = 1.0
    eccentricity: float = 0.0
    # color (unfiltered, measured on the eroded core, relative to field bg)
    teal_fraction: float = 0.0
    relative_blueness: float = 0.0      # (B-R)_core - (B-R)_field
    median_hue: float = 0.0
    median_saturation: float = 0.0
    median_value: float = 0.0
    # classification
    is_stained: bool = False
    stain_confidence: float = 0.0        # signed distance from decision boundary
    # QC flags
    on_border: bool = False
    possible_doublet: bool = False
    is_debris: bool = False
    ambiguous: bool = False
    # provenance / SAM3
    source: str = "watershed"           # "watershed" | "sam3" (how the cell was detected)
    sam3_detect_score: float = 0.0      # SAM3 "cell"-concept score (0 if not from SAM3)
    sam3_stain_score: float = 0.0       # SAM3 "stain"-concept score at this cell (0 if none)

    def as_row(self, decimals: int = 3) -> dict:
        row = {}
        for col in CSV_COLUMNS:
            val = getattr(self, col)
            if isinstance(val, float):
                val = round(val, decimals)
            elif isinstance(val, (bool, np.bool_)):
                val = bool(val)
            elif isinstance(val, (np.integer,)):
                val = int(val)
            row[col] = val
        return row


@dataclass
class SegmentationResult:
    """Output of Stage 1. ``cells`` carries only the geometry fields filled;
    color/classification fields are populated later by Stage 2."""

    labels: np.ndarray                  # int32 label image, 0 = background
    cells: list[CellRecord]
    median_area: float
    foreground_fraction: float
    # intermediate maps for debug/QC (may be None when debug is off)
    background: Optional[np.ndarray] = None
    darkness: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    distance: Optional[np.ndarray] = None
    seeds: Optional[np.ndarray] = None


@dataclass
class AlignmentResult:
    aligned: bool
    dx: float = 0.0
    dy: float = 0.0
    residual_px: float = 0.0
    method: str = "none"
    warped: bool = False


@dataclass
class ClassificationResult:
    """Output of Stage 2. ``cells`` is the same list of :class:`CellRecord`
    objects from segmentation, now with color + classification filled in."""

    cells: list[CellRecord]
    field_bg_bgr: tuple[int, int, int]
    field_br: float                      # field (B - R), the warm-cast reference
    method: str
    teal_mask: Optional[np.ndarray] = None
    threshold_used: float = 0.0
    fallback_used: bool = False          # GMM not separable -> fell back to threshold
