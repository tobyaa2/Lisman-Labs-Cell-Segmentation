"""End-to-end orchestration: filtered + unfiltered -> counts, table, overlay.

Also hosts the calibration helpers (threshold sweep, ground-truth matching) used
by the ``calibrate`` CLI subcommand.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import io_utils, metrics, registration, segmentation, visualization
from .classification import classify
from .config import Config, load_config
from .models import AlignmentResult, CellRecord, ClassificationResult, SegmentationResult


@dataclass
class Result:
    summary: dict
    cells: list[CellRecord]
    labels: np.ndarray
    seg: SegmentationResult
    classification: ClassificationResult
    alignment: AlignmentResult
    config: Config
    unfiltered_aligned: np.ndarray


def run_pipeline(filtered_bgr: np.ndarray, unfiltered_bgr: np.ndarray,
                 cfg: Config, image_id: str = "image",
                 keep_debug: bool = False) -> Result:
    """Run stages 0b–4 on already-loaded images. Pure compute, no disk I/O."""
    cfg.resolve()

    # Stage 0b: alignment.
    unf_aligned, alignment = registration.align(filtered_bgr, unfiltered_bgr, cfg.registration)

    # Stage 1: segmentation on the filtered image.
    seg = segmentation.segment(filtered_bgr, cfg.segmentation, keep_debug=keep_debug)
    labels, cells = seg.labels, seg.cells

    # Stage 1b (optional): SAM3 exemplar segmentation for detection-union +
    # classification. Runs only when method=="sam3" and exemplars/venv resolve;
    # otherwise the pipeline degrades to the classical color path.
    sam3_stained, labels, cells, sam3_report, sam3_warning = _maybe_run_sam3(
        filtered_bgr, unf_aligned, labels, cells, cfg)

    # Stage 2: stain classification on the (aligned) unfiltered image.
    cls = classify(unf_aligned, labels, cells, cfg.classification,
                   sam3_stained=sam3_stained,
                   sam3_score_threshold=cfg.sam3.stain_score_threshold)

    # Stage 3: aggregate.
    summary = metrics.aggregate(cells, cls, seg.median_area,
                                seg.foreground_fraction, alignment, cfg)
    summary["image_id"] = image_id
    summary["parameters"] = cfg.to_dict()
    summary["version"] = cfg.version
    summary["qc"]["count_robustness_band"] = None  # filled by robustness_band() on request
    summary["qc"]["sam3"] = sam3_report
    if sam3_warning:
        summary.setdefault("warnings", []).append(sam3_warning)

    return Result(
        summary=summary, cells=cells, labels=labels, seg=seg,
        classification=cls, alignment=alignment, config=cfg,
        unfiltered_aligned=unf_aligned,
    )


def _maybe_run_sam3(filtered_bgr, unf_aligned, labels, cells, cfg):
    """Run the two SAM3 concept passes (detection + stain) in one subprocess and
    apply the detection-union.

    Returns (sam3_stained|None, labels, cells, report, warning). Any failure is
    caught and degrades to the color path (sam3_stained=None) unless
    ``cfg.sam3.require``.
    """
    report = {"used": False, "exemplar_source": None, "detect_instances": 0,
              "stain_instances": 0, "recovered_cells": 0, "fallback_reason": None}
    if cfg.classification.method != "sam3":
        return None, labels, cells, report, None

    from . import sam3_client, sam3_detection
    try:
        ok, reason = sam3_client.is_available(cfg.sam3)
        if not ok:                                   # skip the color pre-pass if no venv
            raise sam3_client.Sam3Unavailable(reason)
        if cfg.sam3.exemplar_file:
            exemplars = sam3_client.load_exemplars(cfg.sam3.exemplar_file)
            report["exemplar_source"] = "file"
        elif cfg.sam3.auto_exemplars:
            exemplars = _auto_exemplars(unf_aligned, labels, cells, cfg)
            report["exemplar_source"] = "auto"
        else:
            raise sam3_client.Sam3Unavailable(
                "no --exemplars file and --auto-exemplars is off")
        concepts = exemplars["concepts"]
        passes = []
        if sam3_client.CONCEPT_CELL in concepts:
            spec = concepts[sam3_client.CONCEPT_CELL]
            det_img = filtered_bgr if cfg.sam3.detect_on == "filtered" else unf_aligned
            passes.append({"concept": sam3_client.CONCEPT_CELL, "image": det_img,
                           "boxes": spec["boxes"], "labels": spec["labels"]})
        if sam3_client.CONCEPT_STAIN in concepts:
            spec = concepts[sam3_client.CONCEPT_STAIN]
            passes.append({"concept": sam3_client.CONCEPT_STAIN, "image": unf_aligned,
                           "boxes": spec["boxes"], "labels": spec["labels"]})
        results = sam3_client.run_concepts(passes, cfg.sam3)
    except sam3_client.Sam3Unavailable as e:
        if cfg.sam3.require:
            raise
        report["fallback_reason"] = str(e)
        warning = f"SAM3 requested but unavailable ({e}); used color threshold."
        return None, labels, cells, report, warning

    report["used"] = True
    median_area = _median_watershed_area(cells)
    if sam3_client.CONCEPT_CELL in results:
        det = results[sam3_client.CONCEPT_CELL]
        report["detect_instances"] = len(det)
        labels, cells, added = sam3_detection.merge_detections(
            labels, cells, det, cfg, median_area)
        report["recovered_cells"] = added
    stained = results.get(sam3_client.CONCEPT_STAIN)
    if stained is not None:
        report["stain_instances"] = len(stained)
    return stained, labels, cells, report, None


def _median_watershed_area(cells) -> float:
    areas = [c.area_px for c in cells if c.source != "sam3"]
    return float(np.median(areas)) if areas else 0.0


def _auto_exemplars(unf_aligned, labels, cells, cfg) -> dict:
    """Color pre-pass (fill teal_fraction/is_stained) -> auto-pick exemplar boxes."""
    from . import sam3_auto, sam3_client
    from .classification import extract_features, _classify_threshold
    extract_features(unf_aligned, labels, cells, cfg.classification)
    _classify_threshold(cells, cfg.classification)
    exemplars = sam3_auto.build_auto_exemplars(cells, cfg.sam3)
    if not exemplars["concepts"]:
        raise sam3_client.Sam3Unavailable("auto-exemplars: no usable cells to seed from")
    return exemplars


def run_from_paths(filtered_path: str, unfiltered_path: str, cfg: Config,
                   image_id: Optional[str] = None, keep_debug: bool = False) -> Result:
    filt, unf = io_utils.load_pair(filtered_path, unfiltered_path)
    if image_id is None:
        image_id = _derive_image_id(filtered_path)
    return run_pipeline(filt, unf, cfg, image_id=image_id, keep_debug=keep_debug)


def write_outputs(result: Result, out_dir: str, cfg: Config) -> None:
    io_utils.ensure_dir(out_dir)
    io_utils.write_json(os.path.join(out_dir, "summary.json"), result.summary)
    if cfg.output.save_csv:
        io_utils.write_cells_csv(os.path.join(out_dir, "cells.csv"),
                                 result.cells, decimals=cfg.output.csv_float_decimals)
    if cfg.output.save_overlay:
        overlay = visualization.make_overlay(result.unfiltered_aligned, result.labels,
                                             result.cells, result.summary, cfg)
        io_utils.write_image(os.path.join(out_dir, "overlay.png"), overlay)
    if cfg.output.save_debug:
        dbg = io_utils.ensure_dir(os.path.join(out_dir, "debug"))
        panels = visualization.make_debug_panels(result.seg, result.classification.teal_mask)
        for name, img in panels.items():
            io_utils.write_image(os.path.join(dbg, f"{name}.png"), img)
    if cfg.output.save_histograms:
        hist = visualization.make_histograms(result.cells, cfg)
        if hist is not None:
            io_utils.write_image(os.path.join(out_dir, "histograms.png"), hist)


# ---------------------------------------------------------------------- #
# Calibration / robustness helpers
# ---------------------------------------------------------------------- #
def robustness_band(filtered_bgr: np.ndarray, cfg: Config) -> tuple[int, int]:
    """Re-run segmentation across a small parameter grid; report the (min, max)
    total-cell count as an honest uncertainty band."""
    base = copy.deepcopy(cfg.segmentation).resolve()   # don't assume caller resolved
    counts = []
    for md in {max(1, base.min_distance - 4), base.min_distance, base.min_distance + 4}:
        for af in (0.7, 1.0, 1.3):
            seg_cfg = copy.deepcopy(base)
            seg_cfg.min_distance = md
            seg_cfg.min_cell_area = int(round(base.min_cell_area * af))
            seg = segmentation.segment(filtered_bgr, seg_cfg)
            counts.append(len(seg.cells))
    return (min(counts), max(counts))


def threshold_sweep(result: Result,
                    thresholds: Optional[list[float]] = None) -> list[dict]:
    """Sweep stain_threshold over the *already-extracted* teal fractions and
    report the stained count at each. Cheap: re-thresholds, no recompute."""
    if thresholds is None:
        # focus on the low range where the "any teal" decision actually lives
        thresholds = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    teal = np.array([c.teal_fraction for c in result.cells])
    counted_mask = np.array([_counts_toward_total(c, result.config) for c in result.cells])
    out = []
    for thr in thresholds:
        stained = int(np.sum((teal > thr) & counted_mask))
        out.append({"threshold": float(thr), "stained": stained})
    return out


def _counts_toward_total(c: CellRecord, cfg: Config) -> bool:
    if cfg.classification.exclude_debris and c.is_debris:
        return False
    if not cfg.output.count_border_cells and c.on_border:
        return False
    return True


# ---- ground-truth calibration --------------------------------------- #
def _hungarian_pairs(det: np.ndarray, truth_xy: np.ndarray,
                     radius: float) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment of detections to truth points within
    ``radius`` (Hungarian, per plan §8.2). Returns matched (det_idx, truth_idx)."""
    pairs: list[tuple[int, int]] = []
    if not len(det) or not len(truth_xy):
        return pairs
    from scipy.optimize import linear_sum_assignment
    d = np.hypot(det[:, None, 0] - truth_xy[None, :, 0],
                 det[:, None, 1] - truth_xy[None, :, 1])
    big = float(d.max()) + radius + 1.0          # cost for disallowed (> radius) pairs
    cost = np.where(d <= radius, d, big)
    rows, cols = linear_sum_assignment(cost)
    for i, j in zip(rows, cols):
        if d[i, j] <= radius:
            pairs.append((int(i), int(j)))
    return pairs


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3)}


def match_truth(cells: list[CellRecord], truth_xy: np.ndarray,
                radius: float) -> dict:
    """Detection accuracy: optimal-match detected centroids to ground-truth points
    within ``radius`` and report precision/recall/F1."""
    det = np.array([[c.centroid_x, c.centroid_y] for c in cells], dtype=float)
    tp = len(_hungarian_pairs(det, truth_xy, radius))
    return _prf(tp, len(det) - tp, len(truth_xy) - tp)


def classification_scores(cells: list[CellRecord], truth: np.ndarray,
                          radius: float) -> Optional[dict]:
    """Stain-classification accuracy (plan §8.3). ``truth`` is (N,3): x, y, stained.
    On matched detection/truth pairs, compare ``is_stained`` to the truth label and
    report a confusion matrix + classification precision/recall/F1.

    Returns None if the truth has no stained column or nothing matched."""
    if truth.shape[1] < 3 or not len(cells):
        return None
    det = np.array([[c.centroid_x, c.centroid_y] for c in cells], dtype=float)
    pairs = _hungarian_pairs(det, truth[:, :2], radius)
    if not pairs:
        return None
    tp = fp = fn = tn = 0
    for di, ti in pairs:
        pred = bool(cells[di].is_stained)
        gt = bool(truth[ti, 2] > 0.5)
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1
        else:
            tn += 1
    out = _prf(tp, fp, fn)                       # tp/fp/fn of the *stained* class
    out["tn"] = tn
    out["n_matched"] = len(pairs)
    out["confusion"] = {"true_stained_pred_stained": tp,
                        "true_stained_pred_unstained": fn,
                        "true_unstained_pred_stained": fp,
                        "true_unstained_pred_unstained": tn}
    return out


def _derive_image_id(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    for suffix in ("_Filtered", "_filtered", "_Unfiltered", "_unfiltered"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem
