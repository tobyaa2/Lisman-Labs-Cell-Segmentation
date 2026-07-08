import os

import numpy as np
import pytest

from cell_counter import load_config, run_pipeline
from cell_counter.pipeline import threshold_sweep
from tests import synth

# Path to the real example pair (regression test runs only if present).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
FILTERED = os.path.join(_ROOT, "Example Images", "c1_Filtered.png")
UNFILTERED = os.path.join(_ROOT, "Example Images", "c1_Unfiltered.jpg")
HAVE_REAL = os.path.exists(FILTERED) and os.path.exists(UNFILTERED)


def test_synthetic_end_to_end_counts():
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)
    flags = [True, True, True, False, False, False, True, False, True]
    filt = synth.make_filtered(centers, radius=30)
    unf = synth.make_unfiltered(centers, flags, radius=30)
    cfg = load_config(overrides={"segmentation": {"expected_cell_diameter_px": 60}})
    res = run_pipeline(filt, unf, cfg, image_id="synthetic")
    assert res.summary["total_cells"] == 9
    assert res.summary["stained_cells"] == sum(flags)
    # property: unstained == total - stained
    assert res.summary["unstained_cells"] == (
        res.summary["total_cells"] - res.summary["stained_cells"])


def test_brightness_shift_invariance():
    """Total count must be invariant to a pure global brightness shift after
    background normalization."""
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)
    filt = synth.make_filtered(centers, radius=30)
    cfg = load_config(overrides={"segmentation": {"expected_cell_diameter_px": 60}})
    unf = synth.make_unfiltered(centers, [False] * 9, radius=30)
    base = run_pipeline(filt, unf, cfg, image_id="a").summary["total_cells"]
    brighter = np.clip(filt.astype(np.int16) + 30, 0, 255).astype(np.uint8)
    shifted = run_pipeline(brighter, unf, cfg, image_id="b").summary["total_cells"]
    assert base == shifted


def test_threshold_sweep_monotonic():
    centers = synth.grid_centers(n_side=4, spacing=120, margin=90)
    flags = [i % 2 == 0 for i in range(len(centers))]
    filt = synth.make_filtered(centers, radius=28)
    unf = synth.make_unfiltered(centers, flags, radius=28)
    cfg = load_config(overrides={"segmentation": {"expected_cell_diameter_px": 60}})
    res = run_pipeline(filt, unf, cfg)
    sweep = threshold_sweep(res, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])
    counts = [s["stained"] for s in sweep]
    assert counts == sorted(counts, reverse=True)  # non-increasing in threshold


def test_hungarian_matching_beats_greedy():
    """Optimal assignment must recover both pairs where greedy nearest-neighbor
    would steal a shared truth point (plan §8.2)."""
    from cell_counter.models import CellRecord
    from cell_counter.pipeline import match_truth
    cells = [CellRecord(id=1, centroid_x=0, centroid_y=0, area_px=1, equiv_diameter_px=1),
             CellRecord(id=2, centroid_x=2, centroid_y=0, area_px=1, equiv_diameter_px=1)]
    truth = np.array([[-1.4, 0.0], [1.0, 0.0]])
    m = match_truth(cells, truth, radius=1.6)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 0)


def test_classification_scores_confusion():
    from cell_counter.models import CellRecord
    from cell_counter.pipeline import classification_scores
    cells = [CellRecord(id=1, centroid_x=0, centroid_y=0, area_px=1, equiv_diameter_px=1, is_stained=True),
             CellRecord(id=2, centroid_x=10, centroid_y=0, area_px=1, equiv_diameter_px=1, is_stained=False)]
    truth = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 0.0]])   # both correct
    sc = classification_scores(cells, truth, radius=2.0)
    assert sc is not None
    assert sc["confusion"]["true_stained_pred_stained"] == 1
    assert sc["confusion"]["true_unstained_pred_unstained"] == 1
    assert sc["f1"] == 1.0


def test_confluent_field_does_not_crash():
    """A label map covering every pixel (e.g. an external segmenter's output) must
    not crash field-background estimation."""
    from cell_counter.classification import classify
    from cell_counter.config import ClassificationConfig
    from cell_counter.models import CellRecord
    labels = np.ones((40, 40), np.int32)
    cells = [CellRecord(id=1, centroid_x=20, centroid_y=20, area_px=1600, equiv_diameter_px=45)]
    unf = np.full((40, 40, 3), synth.FIELD_BGR, np.uint8)
    res = classify(unf, labels, cells, ClassificationConfig())
    assert res.field_bg_bgr is not None


def test_robustness_band_on_unresolved_config():
    from cell_counter.config import Config
    from cell_counter.pipeline import robustness_band
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)
    filt = synth.make_filtered(centers, radius=30)
    lo, hi = robustness_band(filt, Config())   # Config() has min_distance/area = None
    assert lo <= hi


@pytest.mark.skipif(not HAVE_REAL, reason="example images not present")
def test_real_pair_regression():
    cfg = load_config()
    res = run_pipeline(
        __import__("cv2").imread(FILTERED),
        __import__("cv2").imread(UNFILTERED),
        cfg, image_id="c1")
    total = res.summary["total_cells"]
    stained = res.summary["stained_cells"]
    assert 260 <= total <= 295, f"total {total} outside validated band 260-295"
    # "any visible teal -> stained" default (stain_threshold=0.01): ~197 stained
    assert 175 <= stained <= 220, f"stained {stained} outside expected band 175-220"
    assert res.summary["qc"]["field_background_bgr"] == [96, 137, 152]
