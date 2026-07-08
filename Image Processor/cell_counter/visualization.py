"""Stage 4 — overlays, debug panels, and feature histograms."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import Config
from .models import CellRecord, SegmentationResult

# BGR colors
_STAINED = (0, 220, 0)        # green outline for stained (teal-positive) cells
_UNSTAINED = (0, 140, 255)    # orange outline for unstained cells
_DOUBLET = (0, 0, 255)        # red outline for possible doublets
_DEBRIS = (200, 200, 200)     # gray for debris


def _draw_label_outlines(canvas: np.ndarray, labels: np.ndarray,
                         cells: list[CellRecord], cfg: Config) -> None:
    """Outline each cell using its label boundary (handles touching cells)."""
    from scipy import ndimage as ndi
    slices = ndi.find_objects(labels)
    for c in cells:
        sl = slices[c.id - 1]
        if sl is None:
            continue
        sub = (labels[sl] == c.id).astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cfg.classification.exclude_debris and c.is_debris:
            color = _DEBRIS
        elif c.possible_doublet:
            color = _DOUBLET
        elif c.is_stained:
            color = _STAINED
        else:
            color = _UNSTAINED
        offset = (sl[1].start, sl[0].start)
        cv2.drawContours(canvas, contours, -1, color, 2, offset=offset)


def _draw_cell_numbers(canvas: np.ndarray, cells: list[CellRecord], cfg: Config) -> None:
    """Print each cell's CSV ``id`` (small) at its centroid so the overlay can be
    matched to cells.csv. White text with a thin black outline reads on both the
    dark teal cells and the pale cream background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font size scales with the cell diameter so numbers stay legible-but-small at
    # any magnification (diameter/150 -> ~0.47 at the default 70px cell).
    diameter = float(cfg.segmentation.expected_cell_diameter_px)
    scale = max(0.3, (diameter / 150.0) * cfg.output.cell_label_scale)
    th = max(1, int(round(scale * 2)))
    for c in cells:
        text = str(c.id)
        (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
        org = (int(c.centroid_x - tw / 2), int(c.centroid_y + tht / 2))
        cv2.putText(canvas, text, org, font, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(canvas, text, org, font, scale, (255, 255, 255), th, cv2.LINE_AA)


def make_overlay(unfiltered_bgr: np.ndarray, labels: np.ndarray,
                 cells: list[CellRecord], summary: dict, cfg: Config) -> np.ndarray:
    canvas = unfiltered_bgr.copy()
    _draw_label_outlines(canvas, labels, cells, cfg)
    if cfg.output.number_cells:
        _draw_cell_numbers(canvas, cells, cfg)

    lines = [
        f"Total: {summary['total_cells']}",
        f"Stained: {summary['stained_cells']} ({summary['percent_stained']}%)",
        f"Unstained: {summary['unstained_cells']}",
    ]
    legend = [
        ("stained", _STAINED),
        ("unstained", _UNSTAINED),
        ("possible doublet", _DOUBLET),
    ]
    _draw_panel(canvas, lines, legend)
    return canvas


def _draw_panel(canvas: np.ndarray, lines: list[str],
                legend: list[tuple[str, tuple]]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.0, canvas.shape[1] / 1600.0)
    th = max(2, int(round(scale * 2)))
    pad = int(20 * scale)
    line_h = int(45 * scale)
    n = len(lines) + len(legend)
    box_w = int(560 * scale)
    box_h = pad * 2 + line_h * n
    cv2.rectangle(canvas, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    y = pad + line_h
    for text in lines:
        cv2.putText(canvas, text, (pad * 2, y), font, scale, (255, 255, 255), th, cv2.LINE_AA)
        y += line_h
    for text, color in legend:
        cv2.circle(canvas, (pad * 2 + int(12 * scale), y - int(10 * scale)),
                   int(12 * scale), color, -1)
        cv2.putText(canvas, text, (pad * 2 + int(40 * scale), y), font,
                    scale * 0.8, (255, 255, 255), th, cv2.LINE_AA)
        y += line_h


def _norm8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def make_debug_panels(seg: SegmentationResult, teal_mask: Optional[np.ndarray]) -> dict:
    panels: dict[str, np.ndarray] = {}
    if seg.background is not None:
        panels["background"] = seg.background
    if seg.darkness is not None:
        panels["darkness"] = _norm8(seg.darkness)
    if seg.mask is not None:
        panels["mask"] = seg.mask
    if seg.distance is not None:
        panels["distance"] = _norm8(seg.distance)
    if seg.seeds is not None:
        seeds_vis = (seg.seeds > 0).astype(np.uint8) * 255
        seeds_vis = cv2.dilate(seeds_vis, np.ones((5, 5), np.uint8))
        panels["seeds"] = seeds_vis
    if teal_mask is not None:
        panels["teal_mask"] = (teal_mask.astype(np.uint8)) * 255
    return panels


def make_histograms(cells: list[CellRecord], cfg: Config) -> Optional[np.ndarray]:
    """Render teal-fraction and area histograms (makes threshold choice visual).
    Returns a BGR image, or None if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not cells:
        return None
    teal = np.array([c.teal_fraction for c in cells])
    area = np.array([c.area_px for c in cells])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(teal, bins=30, color="teal")
    axes[0].axvline(cfg.classification.stain_threshold, color="red", ls="--",
                    label=f"thr={cfg.classification.stain_threshold}")
    axes[0].set_title("teal_fraction per cell"); axes[0].legend()
    axes[1].hist(area, bins=30, color="gray")
    axes[1].set_title("area_px per cell")
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
