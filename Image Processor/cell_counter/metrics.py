"""Stage 3 — aggregate per-cell records into headline numbers + QC."""
from __future__ import annotations

import numpy as np

from .config import Config
from .models import AlignmentResult, CellRecord, ClassificationResult


def _counted(cells: list[CellRecord], cfg: Config) -> list[CellRecord]:
    """Cells that count toward the totals, honoring border / debris exclusion."""
    out = []
    for c in cells:
        if cfg.classification.exclude_debris and c.is_debris:
            continue
        if not cfg.output.count_border_cells and c.on_border:
            continue
        out.append(c)
    return out


def aggregate(cells: list[CellRecord], cls: ClassificationResult,
              seg_median_area: float, foreground_fraction: float,
              alignment: AlignmentResult, cfg: Config) -> dict:
    counted = _counted(cells, cfg)
    total = len(counted)
    stained = sum(1 for c in counted if c.is_stained)
    unstained = total - stained
    percent = round(100.0 * stained / total, 1) if total else 0.0

    areas = np.array([c.area_px for c in counted], dtype=float)
    doublets = sum(1 for c in counted if c.possible_doublet)
    border = sum(1 for c in cells if c.on_border)
    debris = sum(1 for c in cells if c.is_debris)

    # Doublet correction: estimate hidden cells in over-sized regions. Reported
    # SEPARATELY -- never silently inflates the primary count.
    corrected_total = total
    if seg_median_area > 0:
        extra = 0
        for c in counted:
            if c.possible_doublet:
                extra += max(0, int(round(c.area_px / seg_median_area)) - 1)
        corrected_total = total + extra

    qc = {
        "possible_doublets": int(doublets),
        "median_cell_area_px": int(np.median(areas)) if len(areas) else 0,
        "mean_cell_area_px": round(float(np.mean(areas)), 1) if len(areas) else 0.0,
        "border_cells": int(border),
        "debris_flagged": int(debris),
        "foreground_fraction": round(float(foreground_fraction), 4),
        "field_background_bgr": list(cls.field_bg_bgr),
        "field_blue_minus_red": round(float(cls.field_br), 1),
        "stain_method": cls.method,
        "stain_threshold_used": round(float(cls.threshold_used), 4),
        "classifier_fallback_used": bool(cls.fallback_used),
        "alignment": {
            "aligned": bool(alignment.aligned),
            "method": alignment.method,
            "residual_px": round(float(alignment.residual_px), 2),
            "dx": round(float(alignment.dx), 2),
            "dy": round(float(alignment.dy), 2),
            "warped": bool(alignment.warped),
        },
        "doublet_corrected_total": int(corrected_total),
    }

    warnings = []
    if foreground_fraction > 0.6:
        warnings.append("Very high foreground fraction (confluent field?) — counts unreliable.")
    if not alignment.aligned:
        warnings.append(
            f"Images may be misaligned (residual {alignment.residual_px:.1f}px, "
            f"method={alignment.method}) — classification may be corrupted.")
    if total == 0:
        warnings.append("No cells detected.")

    return {
        "total_cells": int(total),
        "stained_cells": int(stained),
        "unstained_cells": int(unstained),
        "percent_stained": percent,
        "qc": qc,
        "warnings": warnings,
    }
