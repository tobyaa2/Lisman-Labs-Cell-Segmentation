"""Torch-free driver for the interactive exemplar picker.

Launches the matplotlib GUI (which lives in the SAM3 venv) once per concept via a
subprocess, and returns the drawn boxes as {"boxes": [[x1,y1,x2,y2],...],
"labels": [1/0,...]} (pixel xyxy in the image's frame).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import cv2

from .sam3_client import is_available

_PICK_GUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_pick_gui.py")


class Sam3PickError(Exception):
    pass


def pick_concept(image_bgr, sam3_cfg, prompt: str = "") -> dict:
    ok, reason = is_available(sam3_cfg)
    if not ok:
        raise Sam3PickError(f"SAM3 unavailable for picking exemplars: {reason}")
    with tempfile.TemporaryDirectory(prefix="sam3pick_") as tmp:
        img_path = os.path.join(tmp, "img.png")
        out_path = os.path.join(tmp, "boxes.json")
        if not cv2.imwrite(img_path, image_bgr):
            raise Sam3PickError("failed to write temp image for picking")
        try:
            proc = subprocess.run(
                [sam3_cfg.python_path, _PICK_GUI, img_path, out_path, prompt, sam3_cfg.meta_dir],
                text=True,
            )
        except OSError as e:
            raise Sam3PickError(f"failed to launch picker GUI: {e}")
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise Sam3PickError("picker GUI did not return boxes (cancelled?)")
        with open(out_path) as fh:
            drawn = json.load(fh)
    pos = [list(map(float, b)) for b in drawn.get("pos", [])]
    neg = [list(map(float, b)) for b in drawn.get("neg", [])]
    return {"boxes": pos + neg, "labels": [1] * len(pos) + [0] * len(neg)}
