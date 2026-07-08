"""Configuration for the cell-counting pipeline.

All tunable parameters live here as a nested :class:`Config` dataclass. Every
size/distance default is *derived* from a single knob — ``expected_cell_diameter_px``
— so the same code works at other magnifications: change the diameter and every
kernel and seed distance rescales. Any field left as ``None`` is filled from the
diameter in :meth:`Config.resolve`; an explicit value in the YAML config always
wins.

The defaults reproduce the validated prototype run on the ``c1`` pair
(total ~283, robustness band ~270-295). The stain default is "any visible teal ->
stained" (``stain_threshold = 0.01``), giving ~197 stained; raise the threshold to
require a stronger stain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional

import yaml

__version__ = "0.1.0"


def _odd(n: float) -> int:
    """Round to the nearest odd integer >= 1 (OpenCV kernels must be odd)."""
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


@dataclass
class SegmentationConfig:
    """Stage 1 — total-cell segmentation on the *filtered* image."""

    # The single master knob. Everything below derives from it when left None.
    expected_cell_diameter_px: float = 70.0

    blur_sigma: float = 2.0             # Gaussian pre-blur on the blue channel
    dist_blur_sigma: float = 2.0        # smoothing applied to the distance transform
    peak_rel_thresh: float = 0.30       # seed threshold as a fraction of max distance
    doublet_factor: float = 1.8         # area > factor * median  -> possible doublet
    split_touching: bool = True         # watershed-split touching cells (False = --no-watershed)
    # Detection sensitivity: the foreground threshold is otsu_factor * Otsu(darkness).
    # 1.0 = default (Otsu). LOWER (e.g. 0.6) catches fainter, lower-contrast cells at
    # the cost of more false positives; the field is faint so pushing much below ~0.5
    # floods the image. See README "Faint cells".
    otsu_factor: float = 1.0

    # Derived-from-diameter when None (factors chosen to reproduce the validated run
    # at d=70: closing=101, min_distance=18, min_cell_area=250).
    closing_kernel: Optional[int] = None      # ~1.45 x diameter, forced odd
    open_kernel: int = 5                       # speck removal
    close_kernel: int = 7                       # pinhole fill
    min_distance: Optional[int] = None         # ~0.26 x diameter
    min_cell_area: Optional[int] = None         # ~0.065 x expected disc area

    # factors (exposed so the derivation is transparent / overridable)
    closing_factor: float = 1.44       # 1.44 x 70 -> 101 (the validated kernel)
    min_distance_factor: float = 0.26
    min_area_factor: float = 0.065

    def resolve(self) -> "SegmentationConfig":
        d = float(self.expected_cell_diameter_px)
        expected_area = math.pi * (d / 2.0) ** 2
        if self.closing_kernel is None:
            self.closing_kernel = _odd(self.closing_factor * d)
        else:
            self.closing_kernel = _odd(self.closing_kernel)
        self.open_kernel = _odd(self.open_kernel)
        self.close_kernel = _odd(self.close_kernel)
        if self.min_distance is None:
            self.min_distance = max(1, int(round(self.min_distance_factor * d)))
        if self.min_cell_area is None:
            self.min_cell_area = max(1, int(round(self.min_area_factor * expected_area)))
        return self


@dataclass
class ClassificationConfig:
    """Stage 2 — stain classification on the *unfiltered* image."""

    method: str = "threshold"           # "threshold" (A) or "gmm" (B)
    # "Any visible teal -> stained": a low floor on the teal-pixel fraction. Set
    # just above single-pixel/neighbor-bleed noise (a real faint stain is ~>=1% of
    # the core). Raise it to require a stronger stain; lower it toward 0 for max
    # sensitivity. (Default was 0.20 = "20% of the cell must be teal", which missed
    # most partially-stained cells.)
    stain_threshold: float = 0.01       # teal_fraction cutoff for method A
    core_erosion: int = 2               # px eroded off each cell mask -> "core"
    field_dilation: int = 15            # px the cell set is dilated to define field bg

    # teal precipitate window (OpenCV HSV: H in 0..179). Widened from 45-105/sat>=35
    # to also catch faint (low-saturation) teal.
    teal_hue_lo: int = 40
    teal_hue_hi: int = 110
    teal_sat_floor: int = 20
    # Dark-teal catch (plan §5.3: very dark teal reads as low-saturation, so combine
    # hue AND value): also count dark, blue-leaning pixels as teal. Set
    # dark_teal_value_max = 0 to disable.
    dark_teal_value_max: int = 90       # only pixels darker than this qualify
    dark_teal_bmr_min: int = 8          # ...and with (B - R) >= this (blue-leaning)

    # borderline / ambiguous band: |teal_fraction - threshold| < margin
    ambiguous_margin: float = 0.05
    emit_ambiguous: bool = False        # if True, borderline cells get is_stained=None-like flag

    # shape sanity -> is_debris (filaments, scratches)
    min_solidity: float = 0.70
    max_eccentricity: float = 0.97
    exclude_debris: bool = False        # debris is always flagged; exclude from counts only if True


@dataclass
class RegistrationConfig:
    """Stage 0b — alignment verification / optional warp."""

    enabled: bool = True
    downscale: float = 0.25             # scale used for the cheap NCC alignment check
    max_offset_frac: float = 0.01       # |offset|/dim above which we register
    residual_warn_px: float = 5.0       # warn loudly above this residual
    min_response: float = 0.03          # phase-corr peak confidence below this -> can't
                                        # determine offset (featureless field) -> assume aligned
    method: str = "auto"                # auto | phase | ecc | orb | none


_DEFAULT_SAM3_META_DIR = "/Volumes/USBDevice/Lisman Labs/Meta SAM3"


@dataclass
class Sam3Config:
    """SAM3 exemplar-segmentation backend (optional, heavy, separate venv).

    SAM3 runs in its own Python venv (torch/transformers). This package stays
    torch-free and shells out to ``sam3_bridge.py`` via ``python_path``. Used, when
    enabled and given a per-image exemplar file, to (a) recover faint cells the
    filtered-image detector misses (detection-union) and (b) classify staining.

    Empirically on the c1 pair: model loads ~7s, ~8-10s/pass on MPS; SAM3 caps at
    200 object queries, so on dense fields it can't enumerate every cell -> detection
    is a UNION (only adds) and classification falls back to color for cells SAM3
    doesn't cover.
    """

    enabled: bool = True                 # master switch; if False, never use SAM3
    python_path: str = ""                # abs path to the SAM3 venv python; "" -> auto-detect
    meta_dir: str = ""                   # Meta SAM3 dir (for Sam3.py + hf_cache); "" -> auto-detect
    model_id: str = "facebook/sam3"
    exemplar_file: str = ""              # per-image exemplar JSON; "" -> use auto/off
    # Auto-exemplars: bootstrap SAM3 exemplar boxes from the color classifier's most
    # confident cells (no manual boxing; works in batch). Used when method=="sam3"
    # and no exemplar_file is given.
    auto_exemplars: bool = False
    auto_cell_count: int = 6             # # cell exemplars (size-diverse) for detection
    auto_stain_pos: int = 6              # # positive (clearly teal) stain exemplars
    auto_stain_neg: int = 5              # # negative (clearly cream) stain exemplars
    exemplar_box_factor: float = 0.6     # exemplar box half-width as a fraction of cell diameter
    score_threshold: float = 0.3         # keep SAM3 instances with score >= this
    mask_threshold: float = 0.5          # binarize SAM3 soft masks at this
    detect_on: str = "filtered"          # run the "cell" concept on "filtered" | "unfiltered"
    detection_iou_new: float = 0.15      # a SAM3 cell is NEW if its overlap w/ existing < this
    stain_score_threshold: float = 0.45  # SAM3 stain score >= this -> stained (0.5 ~93% precise)
    require: bool = False                # if True, missing SAM3 is a hard error (no fallback)
    timeout_s: int = 1200                # subprocess timeout

    def resolve(self) -> "Sam3Config":
        import os
        if not self.meta_dir:
            self.meta_dir = _DEFAULT_SAM3_META_DIR
        if not self.python_path:
            cand = os.path.join(self.meta_dir, ".venv", "bin", "python")
            self.python_path = cand if os.path.exists(cand) else ""
        return self


@dataclass
class OutputConfig:
    save_overlay: bool = True
    save_csv: bool = True
    save_debug: bool = False
    save_histograms: bool = False
    count_border_cells: bool = True     # stereology convention if False
    csv_float_decimals: int = 3
    number_cells: bool = True           # print each cell's CSV id on the overlay
    cell_label_scale: float = 1.0       # multiplier on the auto cell-id font size


@dataclass
class Config:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sam3: Sam3Config = field(default_factory=Sam3Config)
    version: str = __version__

    def resolve(self) -> "Config":
        self.segmentation.resolve()
        self.sam3.resolve()
        return self

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


# ---------------------------------------------------------------------- #
# (de)serialization helpers
# ---------------------------------------------------------------------- #
def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def _merge_into(dc: Any, data: dict[str, Any], path: str = "") -> None:
    """Recursively overlay a plain dict onto a (nested) dataclass instance."""
    valid = {f.name for f in fields(dc)}
    for key, val in data.items():
        if key not in valid:
            raise ValueError(f"Unknown config key: '{path}{key}'")
        cur = getattr(dc, key)
        if is_dataclass(cur) and isinstance(val, dict):
            _merge_into(cur, val, path=f"{path}{key}.")
        else:
            setattr(dc, key, val)


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    """Build a :class:`Config`, overlaying a YAML file and/or a dict of overrides.

    Resolution order (lowest -> highest priority): dataclass defaults, YAML file,
    ``overrides`` dict (e.g. from CLI flags). Derived params are filled at the end.
    """
    cfg = Config()
    if path:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must contain a mapping at top level.")
        # tolerate a top-level 'version' key in the file
        data.pop("version", None)
        _merge_into(cfg, data)
    if overrides:
        _merge_into(cfg, overrides)
    return cfg.resolve()
