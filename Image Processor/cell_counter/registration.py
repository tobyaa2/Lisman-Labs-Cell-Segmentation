"""Stage 0b — alignment verification and optional registration.

For the provided pair the two images are already pixel-aligned, so this is a
cheap no-op. But the program must not *assume* that: we verify every run and
register (warp the unfiltered onto the filtered's frame) only if needed, because
misalignment silently corrupts the per-cell color lookup in Stage 2.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import RegistrationConfig
from .models import AlignmentResult


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _edge_float(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return mag


def _resize_to(img: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def verify_alignment(filtered_bgr: np.ndarray, unfiltered_bgr: np.ndarray,
                     cfg: RegistrationConfig) -> AlignmentResult:
    """Cheap check: phase-correlate downscaled edge maps. If the peak offset is
    ~0 the pair is aligned."""
    h, w = filtered_bgr.shape[:2]
    unf = _resize_to(unfiltered_bgr, (h, w))
    s = max(0.05, min(1.0, cfg.downscale))
    fe = _edge_float(_gray(cv2.resize(filtered_bgr, None, fx=s, fy=s)))
    ue = _edge_float(_gray(cv2.resize(unf, None, fx=s, fy=s)))
    win = cv2.createHanningWindow((fe.shape[1], fe.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(fe * win, ue * win)
    # offsets are in the downscaled frame -> back to full-res px
    dx_full, dy_full = dx / s, dy / s
    offset = float(np.hypot(dx_full, dy_full))
    tol = cfg.max_offset_frac * max(h, w)
    # A low phase-correlation response means there is no reliable peak (e.g. a
    # featureless / empty field). We then cannot tell the offset, so we must NOT
    # cry misalignment -- default to aligned (and such fields have no cells anyway).
    confident = resp >= cfg.min_response
    return AlignmentResult(
        aligned=bool(offset <= tol or not confident),
        dx=float(dx_full), dy=float(dy_full),
        residual_px=offset if confident else 0.0,
        method="phase_corr_check",
    )


def _register_phase(filtered_bgr, unf):
    """Translation-only registration via phase correlation of edge maps.

    Edge/gradient maps are modality-robust, so this aligns the dark-on-blue
    filtered image to the warm-cast unfiltered image where intensity-based ECC
    cannot. Recovers pure translation (the common misalignment)."""
    fe = _edge_float(_gray(filtered_bgr))
    ue = _edge_float(_gray(unf))
    win = cv2.createHanningWindow((fe.shape[1], fe.shape[0]), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(fe * win, ue * win)
    h, w = filtered_bgr.shape[:2]
    warp = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(unf, warp, (w, h), flags=cv2.INTER_LINEAR)
    return aligned, warp


def _register_ecc(filtered_bgr, unf, warp_mode=cv2.MOTION_AFFINE):
    # ECC on *gradient-magnitude* images, which are far more robust to the
    # filtered/unfiltered modality difference than raw intensity.
    def _g(img):
        e = _edge_float(_gray(img))
        m = e.max()
        return (e / m) if m > 0 else e
    fg, ug = _g(filtered_bgr), _g(unf)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    _cc, warp = cv2.findTransformECC(fg, ug, warp, warp_mode, criteria, None, 5)
    h, w = filtered_bgr.shape[:2]
    aligned = cv2.warpAffine(unf, warp, (w, h),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, warp


def _register_orb(filtered_bgr, unf):
    fg, ug = _gray(filtered_bgr), _gray(unf)
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(fg, None)
    k2, d2 = orb.detectAndCompute(ug, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        raise RuntimeError("ORB: too few features for registration")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(d2, d1), key=lambda m: m.distance)[:200]
    if len(matches) < 10:
        raise RuntimeError("ORB: too few matches for registration")
    src = np.float32([k2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homog, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homog is None:
        raise RuntimeError("ORB: homography estimation failed")
    h, w = filtered_bgr.shape[:2]
    return cv2.warpPerspective(unf, homog, (w, h), flags=cv2.INTER_LINEAR), homog


def align(filtered_bgr: np.ndarray, unfiltered_bgr: np.ndarray,
          cfg: RegistrationConfig) -> tuple[np.ndarray, AlignmentResult]:
    """Verify alignment; register only if needed. Returns the unfiltered image in
    the filtered image's frame plus an :class:`AlignmentResult` for QC."""
    h, w = filtered_bgr.shape[:2]
    unf = _resize_to(unfiltered_bgr, (h, w))

    if not cfg.enabled or cfg.method == "none":
        return unf, AlignmentResult(aligned=True, method="skipped")

    result = verify_alignment(filtered_bgr, unf, cfg)
    if result.aligned:
        return unf, result

    # Misaligned -> register. Try the configured method, then graceful fallbacks.
    registrars = {"phase": _register_phase, "ecc": _register_ecc, "orb": _register_orb}
    order = {"auto": ["phase", "ecc", "orb"], "phase": ["phase", "ecc", "orb"],
             "ecc": ["ecc", "orb"], "orb": ["orb"]}.get(cfg.method, ["phase", "ecc", "orb"])
    best = None
    for method in order:
        try:
            aligned, _ = registrars[method](filtered_bgr, unf)
            post = verify_alignment(filtered_bgr, aligned, cfg)
            res = AlignmentResult(
                aligned=post.aligned, dx=post.dx, dy=post.dy,
                residual_px=post.residual_px, method=method, warped=True,
            )
            if best is None or res.residual_px < best[1].residual_px:
                best = (aligned, res)
            if post.aligned:                 # good enough, stop early
                return aligned, res
        except (cv2.error, RuntimeError):
            continue
    if best is not None:                     # return the least-bad attempt
        return best

    # All registration attempts failed; return the (resized) image with a warning flag.
    result.method = "failed"
    return unf, result
