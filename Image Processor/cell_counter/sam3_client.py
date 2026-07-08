"""Torch-free client for the SAM3 bridge.

Keeps `cell_counter` free of torch/transformers: it writes images + an exemplar
request to disk, runs ``sam3_bridge.py`` under the SAM3 venv python as a
subprocess, and reads back bit-packed instance masks. If the venv, weights, or
exemplars are missing (or the subprocess fails), it raises :class:`Sam3Unavailable`
so the pipeline can fall back to the classical color path.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import cv2
import numpy as np

_BRIDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_bridge.py")

# Internal concept keys -> the natural-language-ish concept label passed to SAM3.
CONCEPT_CELL = "cell"
CONCEPT_STAIN = "stain"


class Sam3Unavailable(Exception):
    """SAM3 could not be used (missing venv/weights/exemplars, or a run failed)."""


def is_available(cfg) -> tuple[bool, str]:
    if not cfg.enabled:
        return False, "sam3.enabled is False"
    if not cfg.python_path or not os.path.exists(cfg.python_path):
        return False, f"SAM3 venv python not found ({cfg.python_path or 'unset'})"
    if not os.path.exists(_BRIDGE):
        return False, "sam3_bridge.py missing"
    return True, ""


def load_exemplars(path: str) -> dict:
    """Load + validate a per-image exemplar JSON (schema in the pick-exemplars flow).

    {"concepts": {"cell": {"boxes": [[x1,y1,x2,y2],...], "labels": [1,1,0]},
                  "stain": {...}}}  — boxes in pixels, filtered-frame.
    """
    if not path or not os.path.exists(path):
        raise Sam3Unavailable(f"exemplar file not found: {path or 'unset'}")
    with open(path) as fh:
        data = json.load(fh)
    concepts = data.get("concepts", {})
    if not concepts:
        raise Sam3Unavailable(f"exemplar file has no concepts: {path}")
    for name, spec in concepts.items():
        boxes, labels = spec.get("boxes", []), spec.get("labels", [])
        if len(boxes) != len(labels):
            raise Sam3Unavailable(f"concept '{name}': boxes/labels length mismatch")
        if not any(l == 1 for l in labels):
            raise Sam3Unavailable(f"concept '{name}': needs at least one positive (label 1) box")
    return data


def run_concepts(passes: list[dict], cfg) -> dict[str, list[dict]]:
    """Run one or more exemplar passes in a single SAM3 subprocess (model loads once).

    ``passes`` = [{"concept": str, "image": bgr_ndarray, "boxes": [[x1,y1,x2,y2],...],
                   "labels": [1/0,...]}]. Returns {concept: [{"score", "box", "mask"}]}
    with full-resolution bool masks. Raises Sam3Unavailable on any failure.
    """
    ok, reason = is_available(cfg)
    if not ok:
        raise Sam3Unavailable(reason)
    if not passes:
        return {}

    with tempfile.TemporaryDirectory(prefix="sam3_") as tmp:
        req_passes = []
        for i, p in enumerate(passes):
            img_path = os.path.join(tmp, f"img_{i}.png")
            if not cv2.imwrite(img_path, p["image"]):
                raise Sam3Unavailable(f"failed to write temp image {img_path}")
            req_passes.append({
                "concept": p["concept"],
                "image": img_path,
                "boxes": [list(map(float, b)) for b in p["boxes"]],
                "labels": [int(v) for v in p["labels"]],
                "out": os.path.join(tmp, f"out_{i}.npz"),
            })
        request = {
            "meta_dir": cfg.meta_dir,
            "model_id": cfg.model_id,
            "score_threshold": cfg.score_threshold,
            "mask_threshold": cfg.mask_threshold,
            "passes": req_passes,
        }
        req_path = os.path.join(tmp, "request.json")
        with open(req_path, "w") as fh:
            json.dump(request, fh)

        try:
            proc = subprocess.run(
                [cfg.python_path, _BRIDGE, req_path],
                capture_output=True, text=True, timeout=cfg.timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise Sam3Unavailable(f"SAM3 bridge timed out after {cfg.timeout_s}s")
        except OSError as e:
            raise Sam3Unavailable(f"failed to launch SAM3 bridge: {e}")
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            raise Sam3Unavailable("SAM3 bridge failed: " + " | ".join(tail))

        try:
            manifest = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise Sam3Unavailable("SAM3 bridge produced no manifest")

        results: dict[str, list[dict]] = {}
        for pinfo, spec in zip(manifest["passes"], req_passes):
            results[pinfo["concept"]] = _load_instances(spec["out"])
        return results


def _load_instances(npz_path: str) -> list[dict]:
    with np.load(npz_path) as z:
        n = int(z["n"])
        shape = z["shape"]
        scores = z["scores"]
        boxes = z["boxes"]
        if n == 0:
            return []
        K, H, W = int(shape[0]), int(shape[1]), int(shape[2])
        flat = np.unpackbits(z["masks_packed"], axis=1)[:, : H * W]
        masks = flat.reshape(K, H, W).astype(bool)
    return [{"score": float(scores[i]), "box": boxes[i].tolist(), "mask": masks[i]}
            for i in range(n)]
