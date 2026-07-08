"""Auto-exemplars: bootstrap SAM3 exemplar boxes from the color classifier's most
confident cells, so SAM3 needs no manual boxing (and works headless / in batch).

The color pipeline runs first (to fill each cell's ``teal_fraction``/``is_stained``),
then we pick:
  - "cell"  concept: a size-diverse sample of detected cells (all positive) so SAM3
    can find more cells like them (detection-union recovery).
  - "stain" concept: the clearly-teal cells as POSITIVES and the clearly-cream
    (zero-teal) cells as NEGATIVES, teaching SAM3 to separate teal from cream.

Note: exemplars come from cells the color detector *already found*, so auto-mode's
faint-cell recovery is weaker than hand-drawing boxes on faint cells — but it is
fully automatic. Classification exemplars are clean either way.
"""
from __future__ import annotations

from .models import CellRecord


def _box(c: CellRecord, factor: float) -> list[float]:
    r = factor * c.equiv_diameter_px
    return [c.centroid_x - r, c.centroid_y - r, c.centroid_x + r, c.centroid_y + r]


def _spread(items: list, k: int) -> list:
    """Evenly-spaced sample of ``k`` items from a sorted list (for size diversity)."""
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def build_auto_exemplars(cells: list[CellRecord], cfg_sam3) -> dict:
    """Return an exemplar dict {"concepts": {"cell": {...}, "stain": {...}}} built
    from color-classified cells. Concepts with no positive box are omitted."""
    f = cfg_sam3.exemplar_box_factor
    ok = [c for c in cells if not c.on_border and not c.is_debris and c.area_px > 0]
    if not ok:
        ok = [c for c in cells if c.area_px > 0]

    concepts: dict = {}

    # cell concept: size-diverse positives
    by_area = sorted(ok, key=lambda c: c.area_px)
    cell_picks = _spread(by_area, cfg_sam3.auto_cell_count)
    if cell_picks:
        concepts["cell"] = {"boxes": [_box(c, f) for c in cell_picks],
                            "labels": [1] * len(cell_picks)}

    # stain concept: clearly-teal positives + zero-teal negatives
    by_teal = sorted(ok, key=lambda c: c.teal_fraction, reverse=True)
    pos = [c for c in by_teal if c.is_stained and c.teal_fraction > 0][: cfg_sam3.auto_stain_pos]
    zero_teal = [c for c in ok if c.teal_fraction <= 0.0]
    negs = zero_teal[: cfg_sam3.auto_stain_neg] or by_teal[::-1][: cfg_sam3.auto_stain_neg]
    if pos:                                   # SAM3 needs at least one positive
        boxes = [_box(c, f) for c in pos] + [_box(c, f) for c in negs]
        labels = [1] * len(pos) + [0] * len(negs)
        concepts["stain"] = {"boxes": boxes, "labels": labels}

    return {"version": 1, "auto": True, "concepts": concepts}
