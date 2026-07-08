"""Stage 2 — stain classification on the *unfiltered* image.

For each cell detected on the filtered image we look up its color in the
unfiltered image and decide stained (teal / X-gal positive) vs unstained.

Key principle: the unfiltered image has a strong warm cast, so absolute color
thresholds fail. Every color feature is measured **relative to the per-image
field background** (the non-cell pixels). Features are computed on the cell
*core* (mask eroded a few px) to avoid dilution by pale halos.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import ClassificationConfig
from .models import CellRecord, ClassificationResult


def _field_background(labels: np.ndarray, dilation: int) -> np.ndarray:
    """Boolean mask of background pixels comfortably away from any cell."""
    cell = labels > 0
    if dilation > 0:
        cell = ndi.binary_dilation(cell, iterations=int(dilation))
    field = ~cell
    if not field.any():           # dilation swallowed all background
        field = labels == 0
    if not field.any():           # pathological: cells truly cover every pixel ->
        field = np.ones(labels.shape, bool)   # fall back to all pixels (never empty)
    return field


def _core_mask(cell_mask: np.ndarray, erosion: int) -> np.ndarray:
    """Erode the cell mask to its core; fall back to the full mask if erosion
    would leave too few pixels (small cells)."""
    if erosion <= 0:
        return cell_mask
    core = ndi.binary_erosion(cell_mask, iterations=int(erosion))
    if core.sum() < 5:
        return cell_mask
    return core


def extract_features(unfiltered_bgr: np.ndarray, labels: np.ndarray,
                     cells: list[CellRecord], cfg: ClassificationConfig
                     ) -> tuple[tuple[int, int, int], float, np.ndarray]:
    """Fill the color features on each cell in-place. Returns
    (field_bg_bgr, field_BR, teal_mask)."""
    B = unfiltered_bgr[:, :, 0].astype(np.int16)
    G = unfiltered_bgr[:, :, 1].astype(np.int16)
    R = unfiltered_bgr[:, :, 2].astype(np.int16)
    BmR = (B - R).astype(np.int16)

    hsv = cv2.cvtColor(unfiltered_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # A pixel is "teal" if it falls in the teal hue band with enough saturation,
    # OR it is dark and blue-leaning (very dark teal that reads as low-saturation).
    teal_mask = ((H >= cfg.teal_hue_lo) & (H <= cfg.teal_hue_hi)
                 & (S >= cfg.teal_sat_floor))
    if cfg.dark_teal_value_max > 0:
        dark_teal = (V < cfg.dark_teal_value_max) & (BmR >= cfg.dark_teal_bmr_min)
        teal_mask = teal_mask | dark_teal

    field = _field_background(labels, cfg.field_dilation)
    field_bgr = (int(np.median(B[field])), int(np.median(G[field])), int(np.median(R[field])))
    field_br = float(np.median(BmR[field]))

    # Per-cell features, computed on a local crop for speed.
    slices = ndi.find_objects(labels)
    for cell in cells:
        sl = slices[cell.id - 1]
        if sl is None:
            continue
        sub_labels = labels[sl]
        cell_mask = sub_labels == cell.id
        core = _core_mask(cell_mask, cfg.core_erosion)

        teal_sub = teal_mask[sl]
        bmr_sub = BmR[sl]
        h_sub, s_sub, v_sub = H[sl], S[sl], V[sl]

        cell.teal_fraction = float(teal_sub[core].mean())
        cell.relative_blueness = float(np.median(bmr_sub[core]) - field_br)
        cell.median_hue = float(np.median(h_sub[core]))
        cell.median_saturation = float(np.median(s_sub[core]))
        cell.median_value = float(np.median(v_sub[core]))

    return field_bgr, field_br, teal_mask


def _flag_debris(cells: list[CellRecord], cfg: ClassificationConfig) -> None:
    for c in cells:
        c.is_debris = bool(c.solidity < cfg.min_solidity
                           or c.eccentricity > cfg.max_eccentricity)


def _classify_threshold(cells: list[CellRecord], cfg: ClassificationConfig) -> float:
    thr = cfg.stain_threshold
    for c in cells:
        c.is_stained = bool(c.teal_fraction > thr)
        c.stain_confidence = float(c.teal_fraction - thr)
        if cfg.emit_ambiguous:
            c.ambiguous = bool(abs(c.teal_fraction - thr) < cfg.ambiguous_margin)
    return thr


def _classify_gmm(cells: list[CellRecord], cfg: ClassificationConfig) -> tuple[float, bool]:
    """Unsupervised 2-class split. Returns (threshold_proxy, fallback_used).

    Falls back to the calibrated threshold if the two clusters aren't separable
    (e.g. an all-negative field), per the plan.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        _classify_threshold(cells, cfg)
        return cfg.stain_threshold, True

    feats = np.array([[c.teal_fraction, c.relative_blueness, c.median_value]
                      for c in cells], dtype=float)
    if len(feats) < 4:
        _classify_threshold(cells, cfg)
        return cfg.stain_threshold, True

    # standardize so the three features contribute comparably
    mu, sd = feats.mean(0), feats.std(0)
    sd[sd == 0] = 1.0
    z = (feats - mu) / sd
    import warnings
    with warnings.catch_warnings():
        # an unseparable field collapses to 1 cluster -> we detect & fall back below
        warnings.simplefilter("ignore")
        gm = GaussianMixture(n_components=2, covariance_type="full",
                             n_init=3, random_state=0).fit(z)
    post = gm.predict_proba(z)
    comp = gm.predict(z)

    # stained cluster = the one with higher mean teal_fraction
    mean_teal = [feats[comp == k, 0].mean() if (comp == k).any() else -1.0
                 for k in range(2)]
    stained_k = int(np.argmax(mean_teal))
    sep = abs(mean_teal[stained_k] - mean_teal[1 - stained_k])

    # Not separable -> fall back. Either cluster collapsed, or the teal means are
    # within the ambiguous margin of each other (no real two-population structure).
    if sep < max(cfg.ambiguous_margin, 0.05) or (comp == stained_k).all() \
            or (comp != stained_k).all():
        _classify_threshold(cells, cfg)
        return cfg.stain_threshold, True

    for c, p, k in zip(cells, post[:, stained_k], comp):
        c.is_stained = bool(k == stained_k)
        c.stain_confidence = float(p - 0.5)
        if cfg.emit_ambiguous:
            c.ambiguous = bool(abs(p - 0.5) < 0.15)
    return float(mean_teal[stained_k] + mean_teal[1 - stained_k]) / 2.0, False


def _classify_sam3(labels: np.ndarray, cells: list[CellRecord],
                   stain_instances: list[dict], cfg: ClassificationConfig,
                   sam3_score_threshold: float) -> float:
    """Hybrid SAM3 classification. SAM3 caps at 200 instances, so it can't cover
    every cell in a dense field. Cells that overlap a SAM3 "stained" instance are
    decided by its score; cells SAM3 doesn't cover fall back to the color rule.
    """
    # Build a per-pixel max stain score from the SAM3 stained instances.
    score_map = np.zeros(labels.shape, dtype=np.float32)
    for inst in sorted(stain_instances, key=lambda x: x["score"]):   # asc -> max wins
        m = inst["mask"]
        if m.shape == labels.shape:
            np.maximum(score_map, np.where(m, inst["score"], 0.0), out=score_map)

    slices = ndi.find_objects(labels)
    thr = cfg.stain_threshold
    for c in cells:
        sl = slices[c.id - 1]
        if sl is None:
            continue
        cmask = labels[sl] == c.id
        s = float(score_map[sl][cmask].max()) if cmask.any() else 0.0
        c.sam3_stain_score = s
        if s > 0.0:                                  # SAM3 has an opinion here
            c.is_stained = bool(s >= sam3_score_threshold)
            c.stain_confidence = float(s - sam3_score_threshold)
        else:                                        # SAM3 didn't cover this cell -> color
            c.is_stained = bool(c.teal_fraction > thr)
            c.stain_confidence = float(c.teal_fraction - thr)
        if cfg.emit_ambiguous:
            c.ambiguous = bool(abs(c.stain_confidence) < cfg.ambiguous_margin)
    return sam3_score_threshold


def classify(unfiltered_bgr: np.ndarray, labels: np.ndarray,
             cells: list[CellRecord], cfg: ClassificationConfig,
             sam3_stained: list[dict] | None = None,
             sam3_score_threshold: float = 0.45) -> ClassificationResult:
    field_bgr, field_br, teal_mask = extract_features(unfiltered_bgr, labels, cells, cfg)
    _flag_debris(cells, cfg)

    if cfg.method == "sam3":
        if sam3_stained is None:                     # SAM3 unavailable -> color fallback
            threshold_used = _classify_threshold(cells, cfg)
            method, fallback = "sam3->threshold", True
        else:
            threshold_used = _classify_sam3(labels, cells, sam3_stained, cfg,
                                            sam3_score_threshold)
            method, fallback = "sam3", False
    elif cfg.method == "gmm":
        threshold_used, fallback = _classify_gmm(cells, cfg)
        method = "gmm" if not fallback else "gmm->threshold"
    else:
        threshold_used = _classify_threshold(cells, cfg)
        method = "threshold"
        fallback = False

    return ClassificationResult(
        cells=cells,
        field_bg_bgr=field_bgr,
        field_br=field_br,
        method=method,
        teal_mask=teal_mask,
        threshold_used=threshold_used,
        fallback_used=fallback,
    )
