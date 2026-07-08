from cell_counter.config import Sam3Config
from cell_counter.models import CellRecord
from cell_counter.sam3_auto import build_auto_exemplars


def _cell(i, x, y, d=60.0, teal=0.0, stained=False, border=False, debris=False):
    return CellRecord(id=i, centroid_x=x, centroid_y=y, area_px=int(d * d), equiv_diameter_px=d,
                      teal_fraction=teal, is_stained=stained, on_border=border, is_debris=debris)


def _cells():
    cells = []
    # 5 clearly teal (stained), 5 clearly cream (zero teal)
    for i in range(5):
        cells.append(_cell(i + 1, 100 + 50 * i, 100, teal=0.4 + 0.05 * i, stained=True))
    for i in range(5):
        cells.append(_cell(i + 6, 100 + 50 * i, 300, teal=0.0, stained=False))
    return cells


def test_builds_cell_and_stain_concepts():
    cfg = Sam3Config(auto_cell_count=4, auto_stain_pos=3, auto_stain_neg=2)
    ex = build_auto_exemplars(_cells(), cfg)
    con = ex["concepts"]
    assert set(con) == {"cell", "stain"}
    assert len(con["cell"]["boxes"]) == 4
    assert all(l == 1 for l in con["cell"]["labels"])            # cell exemplars all positive
    # stain: 3 positives (label 1) + 2 negatives (label 0)
    assert con["stain"]["labels"].count(1) == 3
    assert con["stain"]["labels"].count(0) == 2


def test_boxes_are_pixel_xyxy_around_centroid():
    cfg = Sam3Config(auto_cell_count=1, auto_stain_pos=1, auto_stain_neg=0, exemplar_box_factor=0.5)
    ex = build_auto_exemplars([_cell(1, 200, 150, d=80, teal=0.5, stained=True)], cfg)
    x1, y1, x2, y2 = ex["concepts"]["cell"]["boxes"][0]
    assert (x1, y1, x2, y2) == (160.0, 110.0, 240.0, 190.0)      # 200±40, 150±40


def test_all_cream_field_yields_no_stain_concept():
    # no stained cells -> stain concept omitted (SAM3 needs a positive); detection still works
    cells = [_cell(i + 1, 100 + 40 * i, 100, teal=0.0, stained=False) for i in range(6)]
    ex = build_auto_exemplars(cells, Sam3Config())
    assert "stain" not in ex["concepts"]
    assert "cell" in ex["concepts"]


def test_border_and_debris_cells_excluded():
    cfg = Sam3Config(auto_cell_count=10, auto_stain_pos=10, auto_stain_neg=10)
    cells = _cells() + [_cell(99, 500, 500, teal=0.9, stained=True, border=True),
                        _cell(100, 550, 550, teal=0.9, stained=True, debris=True)]
    ex = build_auto_exemplars(cells, cfg)
    # the border/debris cells (ids 99,100) must not be chosen despite high teal
    n_boxes = len(ex["concepts"]["cell"]["boxes"]) + len(ex["concepts"]["stain"]["boxes"])
    assert n_boxes <= len(_cells()) * 2
