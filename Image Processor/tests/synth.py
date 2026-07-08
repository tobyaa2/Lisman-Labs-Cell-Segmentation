"""Synthetic image generators for the test suite.

Produces matched filtered/unfiltered pairs with a *known* number of cells and a
known stained/unstained split, so tests can assert exact recovery.
"""
from __future__ import annotations

import cv2
import numpy as np

# Field background of the real unfiltered image (warm cast): BGR.
FIELD_BGR = (96, 137, 152)


def _hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    px = np.uint8([[[h, s, v]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


TEAL_BGR = _hsv_to_bgr(80, 180, 120)    # in the teal window, dark -> "stained"
CREAM_BGR = _hsv_to_bgr(24, 150, 190)   # yellow, bright -> "unstained"


def make_filtered(centers, radius=30, shape=(600, 600), bg_blue=146):
    """Blue field (R=G=0, B=bg_blue) with dark discs at ``centers``."""
    img = np.zeros((*shape, 3), np.uint8)
    img[:, :, 0] = bg_blue
    for (x, y) in centers:
        cv2.circle(img, (int(x), int(y)), radius, (15, 0, 0), -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    return img


def make_unfiltered(centers, stained_flags, radius=30, shape=(600, 600)):
    """Warm field with a teal (stained) or cream (unstained) disc per center."""
    img = np.zeros((*shape, 3), np.uint8)
    img[:, :] = FIELD_BGR
    for (x, y), stained in zip(centers, stained_flags):
        color = TEAL_BGR if stained else CREAM_BGR
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    return img


def grid_centers(n_side=3, spacing=160, margin=120):
    return [(margin + i * spacing, margin + j * spacing)
            for j in range(n_side) for i in range(n_side)]
