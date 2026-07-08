import json
import os

import numpy as np
import pytest

from cell_counter import sam3_client
from cell_counter.config import Sam3Config


def _cfg(tmp_path, python_ok=True):
    cfg = Sam3Config()
    cfg.python_path = str(tmp_path / "python") if python_ok else ""
    if python_ok:
        (tmp_path / "python").write_text("#!/bin/sh\n")
    cfg.meta_dir = str(tmp_path)
    return cfg


def test_unavailable_when_python_missing(tmp_path):
    cfg = _cfg(tmp_path, python_ok=False)
    ok, reason = sam3_client.is_available(cfg)
    assert ok is False
    with pytest.raises(sam3_client.Sam3Unavailable):
        sam3_client.run_concepts([{"concept": "cell", "image": np.zeros((4, 4, 3), np.uint8),
                                   "boxes": [[0, 0, 1, 1]], "labels": [1]}], cfg)


def test_run_concepts_parses_bridge_output(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    H, W = 6, 8
    mask = np.zeros((1, H, W), bool); mask[0, 1:4, 1:4] = True

    def fake_run(argv, **kwargs):
        # argv = [python, bridge, request.json]; write the npz the bridge would produce
        req = json.load(open(argv[2]))
        out = req["passes"][0]["out"]
        packed = np.packbits(mask.reshape(1, -1), axis=1)
        np.savez_compressed(out, masks_packed=packed, n=1,
                            shape=np.array([1, H, W]), scores=np.array([0.77], "float32"),
                            boxes=np.array([[1, 1, 4, 4]], "float32"))
        manifest = {"device": "cpu", "passes": [{"concept": "cell", "n": 1, "out": out, "H": H, "W": W}]}

        class R:
            returncode = 0
            stdout = json.dumps(manifest)
            stderr = ""
        return R()

    monkeypatch.setattr(sam3_client.subprocess, "run", fake_run)
    res = sam3_client.run_concepts(
        [{"concept": "cell", "image": np.zeros((H, W, 3), np.uint8), "boxes": [[0, 0, 1, 1]], "labels": [1]}], cfg)
    assert "cell" in res and len(res["cell"]) == 1
    inst = res["cell"][0]
    assert inst["score"] == pytest.approx(0.77, abs=1e-4)
    assert inst["mask"].shape == (H, W)
    assert np.array_equal(inst["mask"], mask[0])


def test_nonzero_exit_raises(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)

    def fail_run(argv, **kwargs):
        class R:
            returncode = 1
            stdout = ""
            stderr = "boom\ntorch error"
        return R()

    monkeypatch.setattr(sam3_client.subprocess, "run", fail_run)
    with pytest.raises(sam3_client.Sam3Unavailable):
        sam3_client.run_concepts(
            [{"concept": "cell", "image": np.zeros((4, 4, 3), np.uint8), "boxes": [[0, 0, 1, 1]], "labels": [1]}], cfg)


def test_load_exemplars_validates(tmp_path):
    good = tmp_path / "ex.json"
    good.write_text(json.dumps({"concepts": {"cell": {"boxes": [[0, 0, 1, 1]], "labels": [1]}}}))
    data = sam3_client.load_exemplars(str(good))
    assert "cell" in data["concepts"]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"concepts": {"cell": {"boxes": [[0, 0, 1, 1]], "labels": [0]}}}))
    with pytest.raises(sam3_client.Sam3Unavailable):   # no positive box
        sam3_client.load_exemplars(str(bad))

    with pytest.raises(sam3_client.Sam3Unavailable):
        sam3_client.load_exemplars(str(tmp_path / "missing.json"))
