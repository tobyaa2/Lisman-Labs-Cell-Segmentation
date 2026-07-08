import numpy as np
import pytest

from cell_counter.classification import classify
from cell_counter.config import ClassificationConfig, SegmentationConfig
from cell_counter.segmentation import segment
from tests import synth


def _segment(centers, radius=30):
    filt = synth.make_filtered(centers, radius=radius)
    seg = segment(filt, SegmentationConfig(expected_cell_diameter_px=60).resolve())
    return seg


def test_teal_discs_called_stained_cream_unstained():
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)  # 9 discs
    flags = [True, False, True, False, True, False, True, False, True]  # 5 stained
    seg = _segment(centers)
    unf = synth.make_unfiltered(centers, flags, radius=30)
    cls = classify(unf, seg.labels, seg.cells, ClassificationConfig())

    # match each detected cell back to its nearest synthetic center
    got = {}
    for c in seg.cells:
        idx = int(np.argmin([(c.centroid_x - x) ** 2 + (c.centroid_y - y) ** 2
                             for x, y in centers]))
        got[idx] = c.is_stained
    for idx, want in enumerate(flags):
        assert got.get(idx) == want, f"disc {idx}: want stained={want}"
    stained = sum(1 for c in seg.cells if c.is_stained)
    assert stained == 5


def test_field_background_recovered():
    centers = synth.grid_centers(n_side=2, spacing=200, margin=150)
    seg = _segment(centers)
    unf = synth.make_unfiltered(centers, [True, False, True, False], radius=30)
    cls = classify(unf, seg.labels, seg.cells, ClassificationConfig())
    # field bg should be close to the synthetic warm cast (96,137,152)
    for got, want in zip(cls.field_bg_bgr, synth.FIELD_BGR):
        assert abs(got - want) <= 8


def test_all_negative_field_threshold_gives_zero():
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)
    seg = _segment(centers)
    unf = synth.make_unfiltered(centers, [False] * len(centers), radius=30)
    cls = classify(unf, seg.labels, seg.cells, ClassificationConfig(method="threshold"))
    assert sum(c.is_stained for c in seg.cells) == 0


def test_all_negative_field_gmm_falls_back():
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)
    seg = _segment(centers)
    unf = synth.make_unfiltered(centers, [False] * len(centers), radius=30)
    cls = classify(unf, seg.labels, seg.cells, ClassificationConfig(method="gmm"))
    assert cls.fallback_used is True
    assert sum(c.is_stained for c in seg.cells) == 0


def test_gmm_separates_mixed_field():
    centers = synth.grid_centers(n_side=4, spacing=120, margin=90)  # 16 discs
    flags = [i % 2 == 0 for i in range(len(centers))]
    seg = _segment(centers)
    unf = synth.make_unfiltered(centers, flags, radius=28)
    cls = classify(unf, seg.labels, seg.cells, ClassificationConfig(method="gmm"))
    stained = sum(c.is_stained for c in seg.cells)
    # GMM should land near the true 50/50 split
    assert abs(stained - sum(flags)) <= 2


def test_stain_confidence_sign_matches_label():
    centers = synth.grid_centers(n_side=2, spacing=200, margin=150)
    flags = [True, False, True, False]
    seg = _segment(centers)
    unf = synth.make_unfiltered(centers, flags, radius=30)
    classify(unf, seg.labels, seg.cells, ClassificationConfig(method="threshold"))
    for c in seg.cells:
        assert (c.stain_confidence > 0) == c.is_stained
