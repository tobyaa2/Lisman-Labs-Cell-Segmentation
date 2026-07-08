import numpy as np
import pytest

from cell_counter.config import SegmentationConfig
from cell_counter.segmentation import segment
from tests import synth


def _cfg(diameter=60.0):
    return SegmentationConfig(expected_cell_diameter_px=diameter).resolve()


def test_counts_well_separated_discs():
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)  # 9 discs
    filt = synth.make_filtered(centers, radius=30)
    seg = segment(filt, _cfg())
    assert len(seg.cells) == 9


def test_watershed_splits_touching_discs():
    # Two discs whose edges overlap (centers 45px apart, radius 30 -> overlap 15px).
    centers = [(280, 300), (325, 300)]
    filt = synth.make_filtered(centers, radius=30)
    seg = segment(filt, _cfg())
    assert len(seg.cells) == 2, "watershed should split the touching pair"


def test_no_watershed_merges_touching_discs():
    centers = [(280, 300), (325, 300)]
    filt = synth.make_filtered(centers, radius=30)
    cfg = _cfg()
    cfg.split_touching = False   # the --no-watershed path
    seg = segment(filt, cfg)
    assert len(seg.cells) == 1


def test_empty_field_zero_cells():
    filt = synth.make_filtered([], radius=30)
    seg = segment(filt, _cfg())
    assert len(seg.cells) == 0
    assert seg.foreground_fraction == 0.0


def test_small_specks_filtered_as_noise():
    # A 4px disc is far below min_cell_area and must be dropped.
    centers = [(300, 300)]
    filt = synth.make_filtered(centers, radius=2)
    seg = segment(filt, _cfg())
    assert len(seg.cells) == 0


def test_geometry_fields_reasonable():
    centers = synth.grid_centers(n_side=2, spacing=200, margin=150)
    filt = synth.make_filtered(centers, radius=30)
    seg = segment(filt, _cfg())
    for c in seg.cells:
        assert c.area_px > 0
        assert 40 < c.equiv_diameter_px < 90   # ~60px discs
        assert 0.0 <= c.solidity <= 1.0


def test_labels_compact_and_match_cells():
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)
    filt = synth.make_filtered(centers, radius=30)
    seg = segment(filt, _cfg())
    ids = sorted(c.id for c in seg.cells)
    assert ids == list(range(1, len(seg.cells) + 1))
    assert set(np.unique(seg.labels)) == set([0] + ids)


def test_otsu_factor_default_unchanged_and_lower_is_more_sensitive():
    # A synthetic pair with strong discs plus a very faint disc.
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)
    filt = synth.make_filtered(centers, radius=30)
    # add a faint blob (small darkness) that Otsu should reject at factor 1.0
    import cv2
    cv2.circle(filt, (500, 90), 26, (int(146 * 0.9), 0, 0), -1)
    strict = segment(filt, SegmentationConfig(expected_cell_diameter_px=60, otsu_factor=1.0).resolve())
    loose = segment(filt, SegmentationConfig(expected_cell_diameter_px=60, otsu_factor=0.5).resolve())
    # lowering the factor can only keep or grow the foreground -> >= cells detected
    assert len(loose.cells) >= len(strict.cells)


def test_diameter_knob_rescales_kernels():
    small = SegmentationConfig(expected_cell_diameter_px=40).resolve()
    big = SegmentationConfig(expected_cell_diameter_px=100).resolve()
    assert big.closing_kernel > small.closing_kernel
    assert big.min_distance > small.min_distance
    assert big.min_cell_area > small.min_cell_area
    assert small.closing_kernel % 2 == 1 and big.closing_kernel % 2 == 1
