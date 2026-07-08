import numpy as np
import cv2
import pytest

from cell_counter.config import RegistrationConfig
from cell_counter.registration import align, verify_alignment
from tests import synth


def _textured_pair(shape=(600, 600)):
    centers = synth.grid_centers(n_side=4, spacing=130, margin=80)
    flags = [i % 2 == 0 for i in range(len(centers))]
    filt = synth.make_filtered(centers, radius=28, shape=shape)
    unf = synth.make_unfiltered(centers, flags, radius=28, shape=shape)
    return filt, unf


def test_identical_pair_is_aligned():
    filt, unf = _textured_pair()
    res = verify_alignment(filt, unf, RegistrationConfig())
    assert res.aligned is True
    assert res.residual_px < 5


def test_shift_detected_as_misaligned():
    filt, unf = _textured_pair()
    M = np.float32([[1, 0, 20], [0, 1, 14]])
    shifted = cv2.warpAffine(unf, M, (unf.shape[1], unf.shape[0]))
    res = verify_alignment(filt, shifted, RegistrationConfig())
    assert res.aligned is False
    assert res.residual_px > 5


def test_align_recovers_translation():
    filt, unf = _textured_pair()
    M = np.float32([[1, 0, 12], [0, 1, 9]])
    shifted = cv2.warpAffine(unf, M, (unf.shape[1], unf.shape[0]))
    aligned_img, res = align(filt, shifted, RegistrationConfig(method="auto"))
    assert res.warped is True
    assert res.aligned is True
    assert res.residual_px < 5, f"residual after registration too large: {res.residual_px}"


def test_resize_handles_size_mismatch():
    filt, unf = _textured_pair(shape=(600, 600))
    small = cv2.resize(unf, (300, 300))
    aligned_img, res = align(filt, small, RegistrationConfig())
    assert aligned_img.shape[:2] == filt.shape[:2]


def test_featureless_field_not_flagged_misaligned():
    # A blank/empty field has no phase-correlation peak -> we must NOT cry
    # misalignment (regression: low-confidence response was reported as 282px).
    filt = np.zeros((400, 400, 3), np.uint8); filt[:, :, 0] = 146
    unf = np.zeros((400, 400, 3), np.uint8); unf[:] = synth.FIELD_BGR
    res = verify_alignment(filt, unf, RegistrationConfig())
    assert res.aligned is True


def test_disabled_registration_is_noop():
    filt, unf = _textured_pair()
    aligned_img, res = align(filt, unf, RegistrationConfig(enabled=False))
    assert res.method == "skipped"
    assert res.aligned is True
