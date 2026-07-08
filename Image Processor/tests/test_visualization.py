import numpy as np

from cell_counter import load_config
from cell_counter.segmentation import segment
from cell_counter.classification import classify
from cell_counter.visualization import make_overlay
from tests import synth


def _setup(n_side=3):
    centers = synth.grid_centers(n_side=n_side, spacing=160, margin=120)
    flags = [i % 2 == 0 for i in range(len(centers))]
    filt = synth.make_filtered(centers, radius=30)
    unf = synth.make_unfiltered(centers, flags, radius=30)
    cfg = load_config(overrides={"segmentation": {"expected_cell_diameter_px": 60}})
    seg = segment(filt, cfg.segmentation)
    classify(unf, seg.labels, seg.cells, cfg.classification)
    summary = {"total_cells": len(seg.cells), "stained_cells": 0,
               "unstained_cells": len(seg.cells), "percent_stained": 0.0}
    return unf, seg, summary, cfg


def test_overlay_shape_matches_input():
    unf, seg, summary, cfg = _setup()
    ov = make_overlay(unf, seg.labels, seg.cells, summary, cfg)
    assert ov.shape == unf.shape
    assert ov.dtype == np.uint8


def test_numbering_changes_pixels_and_can_be_disabled():
    unf, seg, summary, cfg = _setup()
    cfg.output.number_cells = True
    numbered = make_overlay(unf, seg.labels, seg.cells, summary, cfg)
    cfg.output.number_cells = False
    plain = make_overlay(unf, seg.labels, seg.cells, summary, cfg)
    # numbering must draw extra (white) pixels -> the two overlays differ
    assert np.any(numbered != plain)


def test_numbering_handles_empty_cell_list():
    unf, seg, summary, cfg = _setup()
    # should not raise when there are no cells to number
    make_overlay(unf, seg.labels, [], {"total_cells": 0, "stained_cells": 0,
                                       "unstained_cells": 0, "percent_stained": 0.0}, cfg)
