"""Command-line entry point. Orchestrates the pipeline stages.

Examples
--------
Standard run::

    python -m cell_counter.cli \\
        --filtered c1_Filtered.png --unfiltered c1_Unfiltered.jpg \\
        --out results/c1 --save-overlay --save-csv

Override the stain threshold and dump debug artifacts::

    python -m cell_counter.cli --filtered c1_Filtered.png --unfiltered c1_Unfiltered.jpg \\
        --out results/c1 --stain-threshold 0.25 --save-debug

Calibrate the threshold against a labeled field::

    python -m cell_counter.cli calibrate --filtered f.png --unfiltered u.jpg \\
        --out results/cal --truth labels.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np

from . import io_utils, pipeline
from .config import __version__, load_config


# ---------------------------------------------------------------------- #
# argument parsing
# ---------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--filtered", required=True, help="Filtered image (dark cells on blue field).")
    p.add_argument("--unfiltered", required=True, help="Unfiltered (true-color) image of the same field.")
    p.add_argument("--config", default=None, help="YAML config overriding any default parameter.")
    p.add_argument("--stain-threshold", type=float, default=None, help="Override the stain teal-fraction threshold.")
    p.add_argument("--expected-diameter", type=float, default=None, help="Expected cell diameter (px); rescales kernels.")
    p.add_argument("--otsu-factor", type=float, default=None,
                   help="Detection sensitivity: threshold = otsu-factor x Otsu. <1 catches "
                        "fainter cells (more false positives); default 1.0.")
    p.add_argument("--closing-kernel", type=int, default=None,
                   help="Background-estimation kernel (px, odd). Larger helps large faint cells.")
    p.add_argument("--classifier", choices=["threshold", "gmm", "sam3"], default=None,
                   help="Stain classifier. Default: sam3 when --exemplars given, else threshold.")
    p.add_argument("--exemplars", default=None,
                   help="SAM3 per-image exemplar JSON (enables SAM3 detection+classification).")
    p.add_argument("--auto-exemplars", action="store_true",
                   help="Auto-pick SAM3 exemplars from the color pass (no manual boxing; works headless).")
    p.add_argument("--sam3-python", default=None, help="Path to the Meta SAM3 venv python.")
    p.add_argument("--sam3-score", type=float, default=None, help="SAM3 stain score cutoff for stained (default 0.45).")
    p.add_argument("--detect-on", choices=["filtered", "unfiltered"], default=None,
                   help="Image for the SAM3 'cell' detection pass (default: filtered).")
    p.add_argument("--require-sam3", action="store_true",
                   help="Fail (exit 4) if SAM3 is unavailable instead of falling back to color.")
    p.add_argument("--no-sam3", action="store_true", help="Disable SAM3 even if exemplars are given.")
    p.add_argument("--no-watershed", action="store_true", help="Disable touching-cell splitting.")
    p.add_argument("--no-register", action="store_true", help="Skip alignment verification/registration.")
    p.add_argument("--count-border", dest="count_border", action="store_true", default=None, help="Count border-clipped cells (default).")
    p.add_argument("--exclude-border", dest="count_border", action="store_false", help="Exclude border-clipped cells (stereology).")
    p.add_argument("--exclude-debris", action="store_true", help="Exclude shape-flagged debris from counts.")
    p.add_argument("--save-overlay", action="store_true")
    p.add_argument("--no-cell-numbers", dest="number_cells", action="store_false", default=None,
                   help="Do not print cell id numbers on the overlay.")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--save-debug", action="store_true")
    p.add_argument("--save-histograms", action="store_true")
    p.add_argument("--robustness-band", action="store_true", help="Compute the total-count robustness band (slower).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cell_counter",
        description="Count cells and classify X-gal/SA-beta-gal staining from a filtered+unfiltered image pair.",
    )
    parser.add_argument("--version", action="version", version=f"cell_counter {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run the pipeline on one image pair (default command).")
    _add_common(run)
    run.add_argument("--out", required=True, help="Output directory.")

    cal = sub.add_parser("calibrate", help="Sweep the stain threshold; optionally score against ground truth.")
    _add_common(cal)
    cal.add_argument("--out", default=None, help="Output directory (optional; prints to stdout regardless).")
    cal.add_argument("--truth", default=None, help="CSV of ground-truth points (x,y[,stained]).")
    cal.add_argument("--match-radius", type=float, default=None, help="Match radius (px); default ~0.5x diameter.")

    batch = sub.add_parser("batch", help="Process a folder of *_Filtered/*_Unfiltered pairs.")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--out", required=True)
    batch.add_argument("--config", default=None)
    batch.add_argument("--stain-threshold", type=float, default=None)
    batch.add_argument("--classifier", choices=["threshold", "gmm", "sam3"], default=None)
    batch.add_argument("--auto-exemplars", action="store_true",
                       help="Use SAM3 with auto-picked exemplars per image (no manual boxing).")
    batch.add_argument("--sam3-python", default=None)
    batch.add_argument("--save-overlay", action="store_true")

    pick = sub.add_parser("pick-exemplars",
                          help="Interactively draw SAM3 exemplars (cell + stain) and save a JSON.")
    pick.add_argument("--filtered", required=True)
    pick.add_argument("--unfiltered", required=True)
    pick.add_argument("--out", required=True, help="Output exemplar JSON path.")
    pick.add_argument("--sam3-python", default=None)
    return parser


def _overrides_from_args(args) -> dict:
    """Translate CLI flags into a nested config-override dict."""
    seg, cls, reg, out, sam3 = {}, {}, {}, {}, {}
    if getattr(args, "stain_threshold", None) is not None:
        cls["stain_threshold"] = args.stain_threshold
    # SAM3: an exemplar file makes "sam3" the effective default classifier.
    exemplars = getattr(args, "exemplars", None)
    if exemplars is not None:
        sam3["exemplar_file"] = exemplars
    if getattr(args, "auto_exemplars", False):
        sam3["auto_exemplars"] = True
    if getattr(args, "sam3_python", None) is not None:
        sam3["python_path"] = args.sam3_python
    if getattr(args, "sam3_score", None) is not None:
        sam3["stain_score_threshold"] = args.sam3_score
    if getattr(args, "detect_on", None) is not None:
        sam3["detect_on"] = args.detect_on
    if getattr(args, "require_sam3", False):
        sam3["require"] = True
    if getattr(args, "no_sam3", False):
        sam3["enabled"] = False
    if getattr(args, "classifier", None) is not None:
        cls["method"] = args.classifier
    elif (exemplars or getattr(args, "auto_exemplars", False)) and not getattr(args, "no_sam3", False):
        cls["method"] = "sam3"                       # effective default when exemplars/auto given
    if getattr(args, "expected_diameter", None) is not None:
        seg["expected_cell_diameter_px"] = args.expected_diameter
    if getattr(args, "otsu_factor", None) is not None:
        seg["otsu_factor"] = args.otsu_factor
    if getattr(args, "closing_kernel", None) is not None:
        seg["closing_kernel"] = args.closing_kernel
    if getattr(args, "no_register", False):
        reg["enabled"] = False
    if getattr(args, "exclude_debris", False):
        cls["exclude_debris"] = True
    if getattr(args, "count_border", None) is not None:
        out["count_border_cells"] = args.count_border
    if getattr(args, "number_cells", None) is not None:
        out["number_cells"] = args.number_cells
    for flag, key in (("save_overlay", "save_overlay"), ("save_csv", "save_csv"),
                      ("save_debug", "save_debug"), ("save_histograms", "save_histograms")):
        if getattr(args, flag, False):
            out[key] = True
    ov: dict = {}
    if seg: ov["segmentation"] = seg
    if cls: ov["classification"] = cls
    if reg: ov["registration"] = reg
    if out: ov["output"] = out
    if sam3: ov["sam3"] = sam3
    return ov


def _no_watershed_patch(cfg) -> None:
    """Disable touching-cell splitting (label connected components only)."""
    cfg.segmentation.split_touching = False


# ---------------------------------------------------------------------- #
# commands
# ---------------------------------------------------------------------- #
def cmd_run(args) -> int:
    from .sam3_client import Sam3Unavailable
    try:
        cfg = load_config(args.config, overrides=_overrides_from_args(args))
        if args.no_watershed:
            _no_watershed_patch(cfg)
        keep_debug = cfg.output.save_debug
        result = pipeline.run_from_paths(args.filtered, args.unfiltered, cfg, keep_debug=keep_debug)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Sam3Unavailable as e:
        print(f"ERROR: SAM3 required but unavailable: {e}", file=sys.stderr)
        return 4

    if args.robustness_band:
        filt, _ = io_utils.load_pair(args.filtered, args.unfiltered)
        result.summary["qc"]["count_robustness_band"] = list(pipeline.robustness_band(filt, cfg))

    pipeline.write_outputs(result, args.out, cfg)
    _print_summary(result.summary)

    align = result.summary["qc"]["alignment"]
    if not align["aligned"] and align["residual_px"] > cfg.registration.residual_warn_px:
        print(f"WARNING: large alignment residual ({align['residual_px']}px).", file=sys.stderr)
        return 3
    return 0


def cmd_calibrate(args) -> int:
    try:
        cfg = load_config(args.config, overrides=_overrides_from_args(args))
        result = pipeline.run_from_paths(args.filtered, args.unfiltered, cfg, keep_debug=False)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    sweep = pipeline.threshold_sweep(result)
    report = {"image_id": result.summary["image_id"], "threshold_sweep": sweep,
              "current_threshold": cfg.classification.stain_threshold,
              "current_stained": result.summary["stained_cells"],
              "total_cells": result.summary["total_cells"]}

    if args.truth:
        truth = _load_truth(args.truth)
        # 0.0 is a valid (if degenerate) radius -> only substitute the default when unset
        radius = (args.match_radius if args.match_radius is not None
                  else 0.5 * cfg.segmentation.expected_cell_diameter_px)
        report["detection"] = pipeline.match_truth(result.cells, truth[:, :2], radius)
        cls_scores = pipeline.classification_scores(result.cells, truth, radius)
        if cls_scores is not None:                # only when truth has a stained column
            report["classification"] = cls_scores

    if args.out:
        io_utils.ensure_dir(args.out)
        io_utils.write_json(os.path.join(args.out, "calibration.json"), report)
    print(json.dumps(report, indent=2))
    return 0


def cmd_batch(args) -> int:
    filtered = sorted(glob.glob(os.path.join(args.input_dir, "*_Filtered.*"))
                      + glob.glob(os.path.join(args.input_dir, "*_filtered.*")))
    if not filtered:
        print(f"ERROR: no *_Filtered.* images in {args.input_dir}", file=sys.stderr)
        return 2
    rows = []
    for fpath in filtered:
        upath = _matching_unfiltered(fpath)
        if upath is None:
            print(f"  skip (no unfiltered match): {os.path.basename(fpath)}", file=sys.stderr)
            continue
        image_id = pipeline._derive_image_id(fpath)
        try:
            cfg = load_config(args.config, overrides=_overrides_from_args(args))
            result = pipeline.run_from_paths(fpath, upath, cfg, image_id=image_id)
            cfg.output.save_overlay = bool(args.save_overlay)
            cfg.output.save_csv = True
            pipeline.write_outputs(result, os.path.join(args.out, image_id), cfg)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  skip ({image_id}): {e}", file=sys.stderr)
            continue
        s = result.summary
        rows.append({"image_id": image_id, "total_cells": s["total_cells"],
                     "stained_cells": s["stained_cells"], "unstained_cells": s["unstained_cells"],
                     "percent_stained": s["percent_stained"]})
        print(f"  {image_id}: total={s['total_cells']} stained={s['stained_cells']}")
    io_utils.ensure_dir(args.out)
    _write_batch_csv(os.path.join(args.out, "batch_summary.csv"), rows)
    print(f"Wrote {len(rows)} results to {args.out}/batch_summary.csv")
    return 0


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _matching_unfiltered(filtered_path: str) -> Optional[str]:
    d = os.path.dirname(filtered_path)
    base = os.path.basename(filtered_path)
    stem, _ = os.path.splitext(base)
    for fsuf, usuf in (("_Filtered", "_Unfiltered"), ("_filtered", "_unfiltered")):
        if stem.endswith(fsuf):
            prefix = stem[: -len(fsuf)]
            for cand in glob.glob(os.path.join(d, prefix + usuf + ".*")):
                return cand
    return None


def _load_truth(path: str) -> np.ndarray:
    import csv
    pts = []
    with open(path) as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    start = 0
    if rows and rows[0] and not _is_number(rows[0][0]):   # header row (guard empty first line)
        start = 1
    for r in rows[start:]:
        if len(r) >= 2 and _is_number(r[0]) and _is_number(r[1]):
            x, y = float(r[0]), float(r[1])
            stained = float(r[2]) if len(r) >= 3 and _is_number(r[2]) else 0.0
            pts.append([x, y, stained])
    return np.array(pts, dtype=float) if pts else np.empty((0, 3))


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _write_batch_csv(path: str, rows: list[dict]) -> None:
    import csv
    cols = ["image_id", "total_cells", "stained_cells", "unstained_cells", "percent_stained"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _print_summary(summary: dict) -> None:
    print(f"image_id:        {summary['image_id']}")
    print(f"total_cells:     {summary['total_cells']}")
    print(f"stained_cells:   {summary['stained_cells']}  ({summary['percent_stained']}%)")
    print(f"unstained_cells: {summary['unstained_cells']}")
    band = summary["qc"].get("count_robustness_band")
    if band:
        print(f"robustness band: {band[0]}-{band[1]}")
    for w in summary.get("warnings", []):
        print(f"WARNING: {w}", file=sys.stderr)


def cmd_pick_exemplars(args) -> int:
    """Draw SAM3 exemplars interactively (in the filtered frame) and save a JSON.

    Aligns the pair, then launches the GUI (in the SAM3 venv) once per concept:
    'cell' boxes on the filtered image, 'stain' boxes on the aligned unfiltered.
    """
    from .config import load_config
    from . import registration
    from .sam3_pick import pick_concept, Sam3PickError

    cfg = load_config(None, overrides=_overrides_from_args(args)).resolve()
    try:
        filt, unf = io_utils.load_pair(args.filtered, args.unfiltered)
        unf_aligned, _ = registration.align(filt, unf, cfg.registration)
        cell = pick_concept(filt, cfg.sam3, prompt="Draw boxes on a few CELLS (incl. faint ones)")
        stain = pick_concept(unf_aligned, cfg.sam3,
                             prompt="Draw POSITIVE boxes on teal cells, NEGATIVE ('n') on cream cells")
    except (FileNotFoundError, ValueError, Sam3PickError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    data = {"version": 1, "image_id": pipeline._derive_image_id(args.filtered),
            "concepts": {"cell": cell, "stain": stain}}
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    io_utils.write_json(args.out, data)
    print(f"Wrote exemplars: cell={len(cell['boxes'])} boxes, stain={len(stain['boxes'])} boxes -> {args.out}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    # default to the 'run' subcommand when none is given
    known = {"run", "calibrate", "batch", "pick-exemplars"}
    if not argv or (argv[0] not in known and argv[0] not in ("-h", "--help", "--version")):
        argv = ["run"] + argv
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        return cmd_calibrate(args)
    if args.command == "batch":
        return cmd_batch(args)
    if args.command == "pick-exemplars":
        return cmd_pick_exemplars(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
