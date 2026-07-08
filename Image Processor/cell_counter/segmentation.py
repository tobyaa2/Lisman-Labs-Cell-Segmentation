"""Stage 1 — total-cell segmentation on the *filtered* image.

The filtered image renders every cell as a dark blob on a uniform blue field,
so we can detect *all* cells (stained or not) by background-subtracting the blue
channel and watershedding the result. This is the validated turn-1 pipeline; all
sizes scale with ``expected_cell_diameter_px`` via the config.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

from .config import SegmentationConfig
from .models import CellRecord, SegmentationResult


def _ellipse(size: int) -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _seed_label_image(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Turn an (N,2) array of (row, col) peak coords into a labelled seed image,
    then connected-component label it so adjacent seed pixels share a marker."""
    pts = np.zeros(shape, dtype=np.uint8)
    if len(coords):
        pts[tuple(coords.T)] = 1
    markers, _ = ndi.label(pts)
    return markers


def segment(filtered_bgr: np.ndarray, cfg: SegmentationConfig,
            keep_debug: bool = False) -> SegmentationResult:
    """Detect and split all cells in the filtered image.

    Returns a :class:`SegmentationResult` whose ``cells`` carry geometry only;
    color/classification fields are filled later by Stage 2.
    """
    cfg = cfg.resolve()   # idempotent; fills any None-valued derived params

    # 1. Blue channel + denoise. The filtered image carries information only in B
    #    (R = G = 0), but we guard against a non-zero R/G by taking B explicitly.
    blue = filtered_bgr[:, :, 0]
    sm = cv2.GaussianBlur(blue, (0, 0), cfg.blur_sigma)

    # 2. Local background via grayscale morphological closing with a kernel larger
    #    than one cell -> fills the dark cell holes -> the bright-field background.
    bg = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, _ellipse(cfg.closing_kernel))

    # 3. Darkness map: cells become bright peaks on a flat ~zero field.
    dark = cv2.subtract(bg, sm)

    # 4. Otsu on the (illumination-flattened) darkness response. otsu_factor scales
    #    the threshold to trade precision for recall on faint, low-contrast cells.
    otsu_t, mask = cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cfg.otsu_factor != 1.0:
        thr = max(1.0, cfg.otsu_factor * float(otsu_t))
        _, mask = cv2.threshold(dark, thr, 255, cv2.THRESH_BINARY)

    # 5. Clean up: open (remove specks) then close (fill pinholes).
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(cfg.open_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(cfg.close_kernel))
    mask_bool = mask > 0
    foreground_fraction = float(mask_bool.mean())

    # 6. Split touching cells with a distance-transform watershed (or, when
    #    splitting is disabled, just connected-component label the mask).
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), cfg.dist_blur_sigma)
    markers = np.zeros(dist.shape, dtype=np.int32)
    if cfg.split_touching and dist.max() > 0:
        coords = peak_local_max(
            dist,
            min_distance=int(cfg.min_distance),
            threshold_abs=cfg.peak_rel_thresh * float(dist.max()),
            labels=mask_bool,
            exclude_border=False,   # keep seeds for cells abutting the frame;
                                    # on_border (not seed exclusion) governs counting
        )
        markers = _seed_label_image(coords, dist.shape)

    if markers.max() > 0:
        labels = watershed(-dist, markers, mask=mask_bool).astype(np.int32)
    else:
        # no splitting requested (or no seeds found) -> label connected components
        labels, _ = ndi.label(mask_bool)
        labels = labels.astype(np.int32)

    # 7. Filter regions by area; relabel to a compact, gap-free id space.
    props = regionprops(labels)
    h, w = labels.shape
    kept = [p for p in props if p.area >= cfg.min_cell_area]
    areas = np.array([p.area for p in kept], dtype=float)
    median_area = float(np.median(areas)) if len(areas) else 0.0
    doublet_area = cfg.doublet_factor * median_area if median_area > 0 else np.inf

    new_labels = np.zeros_like(labels)
    cells: list[CellRecord] = []
    for new_id, p in enumerate(sorted(kept, key=lambda r: r.label), start=1):
        new_labels[labels == p.label] = new_id
        minr, minc, maxr, maxc = p.bbox
        on_border = (minr == 0 or minc == 0 or maxr == h or maxc == w)
        cy, cx = p.centroid
        cells.append(CellRecord(
            id=new_id,
            centroid_x=float(cx),
            centroid_y=float(cy),
            area_px=int(p.area),
            equiv_diameter_px=float(p.equivalent_diameter_area),
            solidity=float(p.solidity),
            eccentricity=float(p.eccentricity),
            on_border=bool(on_border),
            possible_doublet=bool(p.area > doublet_area),
        ))

    return SegmentationResult(
        labels=new_labels,
        cells=cells,
        median_area=median_area,
        foreground_fraction=foreground_fraction,
        background=bg if keep_debug else None,
        darkness=dark if keep_debug else None,
        mask=mask if keep_debug else None,
        distance=dist if keep_debug else None,
        seeds=markers if keep_debug else None,
    )
