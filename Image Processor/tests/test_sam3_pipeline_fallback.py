"""The package must still run when method='sam3' but SAM3 is unavailable
(no exemplar file / no venv): it falls back to the color threshold, so totals
match the classical path and the package is usable out of the box."""
from cell_counter import load_config, run_pipeline
from tests import synth


def _pair():
    centers = synth.grid_centers(n_side=3, spacing=160, margin=120)
    flags = [True, False, True, False, True, False, True, False, True]
    filt = synth.make_filtered(centers, radius=30)
    unf = synth.make_unfiltered(centers, flags, radius=30)
    return filt, unf, sum(flags)


def test_sam3_method_without_exemplars_falls_back():
    filt, unf, n_stained = _pair()
    # method=sam3 but exemplar_file empty -> Sam3Unavailable caught -> color path
    cfg = load_config(overrides={"segmentation": {"expected_cell_diameter_px": 60},
                                 "classification": {"method": "sam3"}})
    cfg.sam3.exemplar_file = ""
    res = run_pipeline(filt, unf, cfg, image_id="synthetic")
    assert res.summary["total_cells"] == 9                 # no detection-union
    assert res.summary["stained_cells"] == n_stained       # color threshold result
    assert res.classification.method == "sam3->threshold"
    assert res.classification.fallback_used is True
    assert res.summary["qc"]["sam3"]["used"] is False
    assert res.summary["qc"]["sam3"]["fallback_reason"]


def test_auto_exemplars_flow(monkeypatch):
    """method=sam3 + auto_exemplars: the color pre-pass builds exemplars and the two
    concepts are passed to SAM3 (mocked). No manual exemplar file needed."""
    from cell_counter import sam3_client
    monkeypatch.setattr(sam3_client, "is_available", lambda cfg: (True, ""))
    seen = {}

    def fake_run(passes, cfg):
        seen["concepts"] = sorted(p["concept"] for p in passes)
        return {p["concept"]: [] for p in passes}      # no instances -> color fallback per cell

    monkeypatch.setattr(sam3_client, "run_concepts", fake_run)
    filt, unf, n_stained = _pair()
    cfg = load_config(overrides={"segmentation": {"expected_cell_diameter_px": 60},
                                 "classification": {"method": "sam3"},
                                 "sam3": {"auto_exemplars": True}})
    res = run_pipeline(filt, unf, cfg, image_id="synthetic")
    assert res.summary["qc"]["sam3"]["used"] is True
    assert res.summary["qc"]["sam3"]["exemplar_source"] == "auto"
    assert seen["concepts"] == ["cell", "stain"]        # auto-built both concepts
    # empty SAM3 results -> classification falls back to color per cell
    assert res.summary["stained_cells"] == n_stained


def test_threshold_method_never_touches_sam3():
    filt, unf, _ = _pair()
    cfg = load_config(overrides={"segmentation": {"expected_cell_diameter_px": 60}})
    res = run_pipeline(filt, unf, cfg, image_id="synthetic")
    assert res.classification.method == "threshold"
    assert res.summary["qc"]["sam3"]["used"] is False
