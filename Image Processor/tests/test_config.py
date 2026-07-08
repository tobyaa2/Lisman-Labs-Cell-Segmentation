import os
import tempfile

import pytest

from cell_counter.config import Config, load_config


def test_defaults_resolve_to_validated_values():
    cfg = load_config()
    seg = cfg.segmentation
    assert seg.closing_kernel == 101      # 1.44 * 70
    assert seg.min_distance == 18         # 0.26 * 70
    assert seg.min_cell_area == 250       # 0.065 * pi * 35^2
    assert seg.closing_kernel % 2 == 1


def test_yaml_override_merges():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("classification:\n  stain_threshold: 0.33\n")
        path = fh.name
    try:
        cfg = load_config(path)
        assert cfg.classification.stain_threshold == 0.33
        assert cfg.segmentation.expected_cell_diameter_px == 70.0  # untouched
    finally:
        os.unlink(path)


def test_overrides_dict_wins_over_yaml():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("classification:\n  stain_threshold: 0.33\n")
        path = fh.name
    try:
        cfg = load_config(path, overrides={"classification": {"stain_threshold": 0.5}})
        assert cfg.classification.stain_threshold == 0.5
    finally:
        os.unlink(path)


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        load_config(overrides={"classification": {"not_a_key": 1}})


def test_explicit_kernel_forced_odd():
    cfg = load_config(overrides={"segmentation": {"closing_kernel": 50}})
    assert cfg.segmentation.closing_kernel == 51


def test_sam3_config_roundtrips_and_rejects_unknown():
    cfg = load_config(overrides={"sam3": {"stain_score_threshold": 0.6, "detect_on": "unfiltered"}})
    assert cfg.sam3.stain_score_threshold == 0.6
    assert cfg.sam3.detect_on == "unfiltered"
    # every sam3 field must survive serialization into summary["parameters"]
    assert cfg.to_dict()["sam3"]["stain_score_threshold"] == 0.6
    with pytest.raises(ValueError):
        load_config(overrides={"sam3": {"not_a_key": 1}})
