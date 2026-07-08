"""Image loading/validation and result writing."""
from __future__ import annotations

import csv
import json
import os
from typing import Any

import cv2
import numpy as np

from .models import CSV_COLUMNS, CellRecord


def load_image(path: str) -> np.ndarray:
    """Read an image as 3-channel BGR uint8. Raises on unreadable/empty files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Could not read image (unsupported/corrupt?): {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel image, got shape {img.shape}: {path}")
    return img


def load_pair(filtered_path: str, unfiltered_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the filtered + unfiltered pair. Does NOT resize — alignment (which may
    resize) is handled in the registration stage."""
    filt = load_image(filtered_path)
    unf = load_image(unfiltered_path)
    return filt, unf


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def write_cells_csv(path: str, cells: list[CellRecord], decimals: int = 3) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for cell in cells:
            writer.writerow(cell.as_row(decimals=decimals))


def write_image(path: str, img: np.ndarray) -> None:
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")
