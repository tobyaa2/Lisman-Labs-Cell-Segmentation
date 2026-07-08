import numpy as np
import pytest

from cell_counter.classification import classify
from cell_counter.config import ClassificationConfig, SegmentationConfig
from cell_counter.segmentation import segment
from tests import synth


def _segment(centers, flags):
    filt = synth.make_filtered(centers, radius=30)
    unf = synth.make_unfiltered(centers, flags, radius=30)
    seg = segment(filt, SegmentationConfig(expected_cell_diameter_px=60).resolve())
    return seg, unf


def _mask_over(labels, cell_id):
    return labels == cell_id


def test_sam3_covered_cell_uses_score():
    centers = synth.grid_centers(n_side=2, spacing=200, margin=150)
    seg, unf = _segment(centers, [False, False, False, False])  # all cream by color
    c = seg.cells[0]
    # SAM3 says cell c is stained with high score, even though color says cream
    inst = [{"score": 0.8, "box": [0, 0, 1, 1], "mask": _mask_over(seg.labels, c.id)}]
    classify(unf, seg.labels, seg.cells, ClassificationConfig(method="sam3"),
             sam3_stained=inst, sam3_score_threshold=0.45)
    assert c.is_stained is True
    assert c.sam3_stain_score == pytest.approx(0.8, abs=1e-5)
    assert c.stain_confidence > 0


def test_sam3_uncovered_cell_falls_back_to_color():
    centers = synth.grid_centers(n_side=2, spacing=200, margin=150)
    seg, unf = _segment(centers, [True, False, False, False])  # cell 0 is teal by color
    # SAM3 covers nobody -> every cell decided by color
    classify(unf, seg.labels, seg.cells, ClassificationConfig(method="sam3"),
             sam3_stained=[], sam3_score_threshold=0.45)
    by_color = {}
    for c in seg.cells:
        idx = int(np.argmin([(c.centroid_x - x) ** 2 + (c.centroid_y - y) ** 2 for x, y in centers]))
        by_color[idx] = c.is_stained
    assert by_color[0] is True                    # the teal disc, via color fallback
    assert sum(by_color.values()) == 1


def test_sam3_low_score_is_unstained():
    centers = synth.grid_centers(n_side=2, spacing=200, margin=150)
    seg, unf = _segment(centers, [False, False, False, False])
    c = seg.cells[0]
    inst = [{"score": 0.30, "box": [0, 0, 1, 1], "mask": _mask_over(seg.labels, c.id)}]
    classify(unf, seg.labels, seg.cells, ClassificationConfig(method="sam3"),
             sam3_stained=inst, sam3_score_threshold=0.45)
    assert c.is_stained is False                  # 0.30 < 0.45
    assert c.stain_confidence < 0


def test_sam3_none_falls_back_to_threshold():
    centers = synth.grid_centers(n_side=2, spacing=200, margin=150)
    seg, unf = _segment(centers, [True, False, True, False])
    res = classify(unf, seg.labels, seg.cells, ClassificationConfig(method="sam3"),
                   sam3_stained=None)
    assert res.method == "sam3->threshold"
    assert res.fallback_used is True
    assert sum(c.is_stained for c in seg.cells) == 2   # color threshold result
