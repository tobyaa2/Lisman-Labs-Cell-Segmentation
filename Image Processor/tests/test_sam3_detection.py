import numpy as np

from cell_counter import load_config
from cell_counter.segmentation import segment
from cell_counter.sam3_detection import merge_detections
from tests import synth


def _setup():
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)
    filt = synth.make_filtered(centers, radius=30)
    cfg = load_config(overrides={"segmentation": {"expected_cell_diameter_px": 60}})
    seg = segment(filt, cfg.segmentation)
    return cfg, seg, seg.median_area


def _disc_mask(shape, cx, cy, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def test_new_cell_in_empty_region_added():
    cfg, seg, med = _setup()
    n0 = len(seg.cells)
    # a disc far from every existing cell (empty corner of the 600x600 frame)
    inst = [{"score": 0.9, "box": [500, 500, 560, 560], "mask": _disc_mask(seg.labels.shape, 530, 530, 28)}]
    labels, cells, added = merge_detections(seg.labels, seg.cells, inst, cfg, med)
    assert added == 1
    assert len(cells) == n0 + 1
    assert cells[-1].source == "sam3"
    assert cells[-1].sam3_detect_score == 0.9
    assert labels.max() == n0 + 1                 # a fresh, gap-free id


def test_overlapping_instance_is_deduped():
    cfg, seg, med = _setup()
    n0 = len(seg.cells)
    c = seg.cells[0]
    inst = [{"score": 0.9, "box": [0, 0, 1, 1],
             "mask": _disc_mask(seg.labels.shape, c.centroid_x, c.centroid_y, 30)}]
    labels, cells, added = merge_detections(seg.labels, seg.cells, inst, cfg, med)
    assert added == 0                             # overlaps an existing cell -> skipped
    assert len(cells) == n0


def test_existing_labels_never_overwritten():
    cfg, seg, med = _setup()
    before = seg.labels.copy()
    inst = [{"score": 0.8, "box": [500, 500, 560, 560], "mask": _disc_mask(seg.labels.shape, 530, 530, 28)}]
    labels, cells, added = merge_detections(seg.labels, seg.cells, inst, cfg, med)
    # every previously-labelled pixel keeps its id
    prev = before > 0
    assert np.array_equal(labels[prev], before[prev])


def test_speck_rejected():
    cfg, seg, med = _setup()
    inst = [{"score": 0.9, "box": [500, 500, 505, 505], "mask": _disc_mask(seg.labels.shape, 530, 530, 3)}]
    labels, cells, added = merge_detections(seg.labels, seg.cells, inst, cfg, med)
    assert added == 0                             # below min_cell_area
