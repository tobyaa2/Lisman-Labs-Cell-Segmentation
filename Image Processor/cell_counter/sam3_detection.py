"""SAM3 detection-union: add cells SAM3 found that the watershed detector missed.

This ONLY adds cells; it never renumbers or removes validated watershed cells, so
the classical count is preserved and recovered faint cells push the total up. New
cells carry ``source="sam3"``; filaments/scratches SAM3 picks up are left for the
existing shape-sanity debris flagging in ``classification._flag_debris``.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import regionprops

from .models import CellRecord


def _record_from_region(region_mask: np.ndarray, new_id: int, median_area: float,
                        doublet_factor: float, score: float) -> CellRecord | None:
    """Build a CellRecord from a boolean region (already placed in the full frame)."""
    ys, xs = np.nonzero(region_mask)
    if len(xs) == 0:
        return None
    minr, maxr, minc, maxc = ys.min(), ys.max(), xs.min(), xs.max()
    crop = region_mask[minr:maxr + 1, minc:maxc + 1].astype(np.uint8)
    props = regionprops(crop)
    if not props:
        return None
    p = props[0]
    H, W = region_mask.shape
    on_border = (minr == 0 or minc == 0 or maxr == H - 1 or maxc == W - 1)
    doublet_area = doublet_factor * median_area if median_area > 0 else np.inf
    return CellRecord(
        id=new_id,
        centroid_x=float(xs.mean()),
        centroid_y=float(ys.mean()),
        area_px=int(region_mask.sum()),
        equiv_diameter_px=float(p.equivalent_diameter_area),
        solidity=float(p.solidity),
        eccentricity=float(p.eccentricity),
        on_border=bool(on_border),
        possible_doublet=bool(region_mask.sum() > doublet_area),
        source="sam3",
        sam3_detect_score=float(score),
    )


def merge_detections(labels: np.ndarray, cells: list[CellRecord],
                     instances: list[dict], cfg, median_area: float
                     ) -> tuple[np.ndarray, list[CellRecord], int]:
    """Union SAM3 "cell" instances into (labels, cells). Returns (labels, cells, n_added).

    An instance is a duplicate (skipped) if it overlaps existing cells by more than
    ``cfg.sam3.detection_iou_new`` of its own area; otherwise it becomes a new cell
    painted into the free pixels of ``labels``.
    """
    min_area = cfg.segmentation.min_cell_area or 1
    doublet_factor = cfg.segmentation.doublet_factor
    iou_new = cfg.sam3.detection_iou_new

    labels = labels.copy()
    occupied = labels > 0
    next_id = int(labels.max())
    added = 0

    for inst in sorted(instances, key=lambda x: -x["score"]):
        m = inst["mask"]
        if m.shape != labels.shape:
            continue
        area = int(m.sum())
        if area < min_area:
            continue
        inter = int(np.logical_and(m, occupied).sum())
        if area > 0 and inter / area >= iou_new:
            continue                                   # duplicate of an existing cell
        new_region = np.logical_and(m, ~occupied)
        if int(new_region.sum()) < min_area:
            continue                                   # what's left after de-overlap is noise
        next_id += 1
        labels[new_region] = next_id
        occupied |= m
        rec = _record_from_region(new_region, next_id, median_area, doublet_factor,
                                  inst["score"])
        if rec is None:
            continue
        cells.append(rec)
        added += 1

    return labels, cells, added
