# cell_counter

Count cells and classify **X‑gal / SA‑β‑gal** staining from a *matched pair* of
microscopy images of the same field — a **filtered** image (dark cells on a
uniform blue field) and an **unfiltered** true‑color image. It reports:

1. **Total cell count** — every cell, stained or not (segmented on the *filtered* image).
2. **Stained cell count** — teal / X‑gal‑positive cells (color read from the *unfiltered* image).

The design rationale lives in [`cell_counter_plan.md`](cell_counter_plan.md). The
core idea: **detect once on the filtered image, then look up color at each cell's
location in the unfiltered image.** Color is always measured *relative to the
per‑image field background* because the unfiltered image has a strong warm cast,
so absolute color thresholds fail.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. `scikit-learn` (GMM classifier) and `matplotlib` (feature histograms)
are optional; everything else works without them. The optional **SAM3 backend**
(faint‑cell recovery + exemplar classification) is heavier and runs in its **own
separate venv** — it is *not* installed by `requirements.txt`; see [SAM3](#sam3-backend-recover-faint-cells--exemplar-driven-classification).

## Quick start

Run the **SAM3 backend** with **`--auto-exemplars`** — fully automatic, **no manual
boxing**. SAM3 bootstraps its example boxes from the color pass:

```bash
python3 -m cell_counter.cli run \
    --filtered   "Example Images/c1_Filtered.png" \
    --unfiltered "Example Images/c1_Unfiltered.jpg" \
    --auto-exemplars \
    --out        results/c1 \
    --save-overlay --save-csv
```

`--auto-exemplars` makes `sam3` the classifier automatically. On `c1` this reports
**total ≈ 295** (283 classical + ~12 SAM3‑recovered cells) and **stained ≈ 193**.

SAM3 runs in its **own venv** (see [Install](#install) and
[SAM3](#sam3-backend-recover-faint-cells--exemplar-driven-classification)). Without
that venv the *same command* transparently falls back to the classical color pipeline:
**total ≈ 283** (robustness band ≈ 270–295), **stained ≈ 197** at the default
`stain_threshold = 0.01` (the "any visible teal → stained" rule; raise
`--stain-threshold` to require a stronger stain). `--robustness-band` reports the
honest total‑count uncertainty.

**For the best faint‑cell recovery,** hand‑draw exemplars *on* the faint cells instead
of auto‑picking — auto seeds from already‑detected cells, so it can miss the faintest.
The bundled `c1_exemplars.json` does this (→ total ≈ 298, stained ≈ 205); for a new
pair, draw your own (GUI: box a few CELLS incl. faint ones on the filtered image, then
POSITIVE teal / NEGATIVE cream boxes — press `n` to toggle — on the unfiltered):

```bash
python3 -m cell_counter.cli pick-exemplars \
    --filtered new_Filtered.png --unfiltered new_Unfiltered.jpg --out exemplars/new.json
python3 -m cell_counter.cli run --filtered new_Filtered.png --unfiltered new_Unfiltered.jpg \
    --exemplars exemplars/new.json --out results/new --save-overlay --save-csv
```

### Common flags

Run `python -m cell_counter.cli run --help` for the complete list. The most useful:

| Flag | Effect |
|---|---|
| `--stain-threshold 0.05` | Raise the teal‑fraction cutoff to require a stronger stain. |
| `--classifier gmm` | Unsupervised 2‑class split (auto‑adapts per image; falls back to the threshold if not separable). |
| `--classifier sam3 --exemplars f.json` | Use the SAM3 backend (see below). |
| `--expected-diameter 90` | Rescale every kernel/distance for a different magnification. |
| `--otsu-factor 0.7` | Detection sensitivity; `<1` catches fainter cells (see [Faint cells](#faint-low-contrast-cells)). |
| `--exclude-border` | Drop cells clipped at the image edge (stereology convention). |
| `--exclude-debris` | Drop shape‑flagged filaments/scratches from the counts. |
| `--no-cell-numbers` | Don't print cell ids on the overlay. |
| `--save-debug` | Dump the background/darkness/mask/distance/seeds/teal‑mask maps to `debug/`. |
| `--save-histograms` | Save per‑cell teal‑fraction and area histograms. |
| `--robustness-band` | Report the total‑count uncertainty band over a small parameter grid. |
| `--no-watershed` | Disable touching‑cell splitting. |
| `--config my.yaml` | Override any parameter (see `configs/default.yaml`). |

## Outputs (written to `--out`)

- **`summary.json`** — headline counts, QC roll‑ups (doublets, field background, alignment, `sam3`), and the full resolved parameter set.
- **`cells.csv`** — one row per cell (schema in [`cell_counter/models.py`](cell_counter/models.py); includes `is_stained`, `stain_confidence`, `source`, and geometry).
- **`overlay.png`** — unfiltered image with stained/unstained/doublet cells outlined, each labeled with its `cells.csv` **id** (disable with `--no-cell-numbers`), and counts printed.
- **`debug/`**, **`histograms.png`** — optional diagnostics (with `--save-debug` / `--save-histograms`).

## Faint (low-contrast) cells

Total segmentation runs on the *filtered* image, where each cell is a dark blob on
a uniform blue field. Some cells are **very pale** — they barely darken the blue
field (measured darkness ~15–45 vs a detection threshold of ~89) and, in the
unfiltered image, sit within ~1–3 color units of the tan background. Such cells are
near the noise floor of **both** images: a fixed threshold low enough to catch them
also turns 40–70% of the frame into false foreground.

To trade precision for recall, lower the Otsu factor (and optionally enlarge the
background kernel):

```bash
python -m cell_counter.cli run --filtered f.png --unfiltered u.jpg --out out \
    --otsu-factor 0.7 --closing-kernel 151      # catches more faint cells
```

| `--otsu-factor` | Behavior |
|---|---|
| `1.0` (default) | Otsu threshold; best precision; misses the faintest cells. |
| `0.7` | Recovers moderately faint cells; ~+25% total (some false positives). |
| `<0.5` | Floods the image — not recommended for this field. |

For **reliably** detecting the faintest cells (which have a shape a human sees but
no threshold can isolate), use the **SAM3 backend** below — a learned segmenter that
recognizes cell morphology rather than raw contrast.

## SAM3 backend (recover faint cells + exemplar-driven classification)

An optional **Meta SAM3** backend (exemplar/concept segmentation) can (a) **recover
faint cells** the filtered‑image detector misses and (b) classify staining. SAM3 runs
in its **own venv** (torch / transformers, ~3.2 GB model); this package stays
torch‑free and calls it through a subprocess bridge (`sam3_client.py` → `sam3_bridge.py`).

SAM3 needs example boxes ("exemplars") that define the concepts. You get them **two
ways** — pick one:

**A. Auto‑exemplars (no boxing, recommended, works headless / in `batch`).** SAM3
bootstraps its exemplars from the color pass — a size‑diverse sample of detected cells
for the "cell" concept, and the clearly‑teal cells (positives) + zero‑teal cells
(negatives) for the "stain" concept:

```bash
python -m cell_counter.cli run --filtered f.png --unfiltered u.jpg \
    --auto-exemplars --out results/c1 --save-overlay --save-csv
```

Because auto seeds from cells the color detector *already found*, its faint‑cell
recovery is modest (on c1: ~12 recovered) and classification mostly matches the color
rule. It requires zero interaction.

**B. Manual exemplars (best faint recovery).** Draw boxes once per image — this is the
way to recover *specific* faint cells (box them directly):

```bash
# Draw a few CELL boxes (include faint ones) on the filtered image, then POSITIVE
# (teal) / NEGATIVE (cream, press 'n') boxes on the unfiltered. Needs a display.
python -m cell_counter.cli pick-exemplars \
    --filtered c1_Filtered.png --unfiltered c1_Unfiltered.jpg --out exemplars/c1.json
python -m cell_counter.cli run --filtered c1_Filtered.png --unfiltered c1_Unfiltered.jpg \
    --exemplars exemplars/c1.json --out results/c1_sam3 --save-overlay --save-csv
```

Either `--auto-exemplars` or an `--exemplars` file makes `sam3` the classifier
automatically.

Detection is a **union**: SAM3‑recovered cells are *added* to the validated
watershed cells (never replacing them), tagged `source=sam3` in `cells.csv`.
Classification is **hybrid**: cells a SAM3 "stained" instance covers are decided by
its score (`--sam3-score`, default 0.45); cells SAM3 doesn't cover fall back to the
color rule. QC (`summary.json` → `qc.sam3`) reports instances found and cells
recovered.

**Honest limits (measured on c1):** SAM3 loads ~7 s and runs ~8–10 s/pass on Apple
MPS. It caps at **200 object queries**, so on a dense 283‑cell field it can't
enumerate every cell — hence detection‑union (only adds) and the color fallback for
uncovered cells. Faint recovery works only if you draw exemplars *on* faint cells,
and SAM3 also grabs filaments/debris (cleaned up by the existing `is_debris` shape
filter). On c1 it recovered the specific faint cells that classical thresholding
misses, taking the total 283 → ~298. SAM3's stain score discriminates teal vs cream
(≈93 % precise at score ≥ 0.5), but the color rule already classifies well — so
SAM3's biggest win here is **detection**, not classification.

**Flags:** `--auto-exemplars`, `--exemplars PATH`, `--classifier sam3`,
`--sam3-score FLOAT`, `--detect-on {filtered,unfiltered}`, `--sam3-python PATH`,
`--require-sam3` (exit 4 instead of falling back), `--no-sam3`. Without a reachable
SAM3 venv or exemplars, the pipeline transparently uses the color path — so it always
runs out of the box. (Auto‑exemplar counts are tunable in `configs/default.yaml`
under `sam3:`.)

## Calibration

The **stained** count is threshold‑driven — on c1 it runs from ≈111 (at
`stain_threshold = 0.30`) up to ≈208 (at `0.0`), and is ≈197 at the default `0.01`.
So calibration matters. The `calibrate` subcommand sweeps the threshold and, if you
supply hand labels, scores accuracy against them:

```bash
python -m cell_counter.cli calibrate \
    --filtered f.png --unfiltered u.jpg --out results/cal \
    --truth labels.csv          # CSV: x,y[,stained] per ground-truth cell
```

It prints (and, with `--out`, writes `calibration.json`) containing:

- **`threshold_sweep`** — stained count at each threshold (to pick an operating point).
- **`detection`** — precision / recall / F1 of detected centroids vs the truth points (optimal Hungarian matching within `--match-radius`).
- **`classification`** — if the truth CSV has a `stained` column: a stained‑vs‑unstained confusion matrix + F1 on the matched cells.

For comparing conditions, keep the threshold **fixed** across all images in an
experiment (consistency matters more than per‑image perfection).

## Batch mode

Point at a folder of `*_Filtered`/`*_Unfiltered` pairs to get per‑image outputs plus
one aggregated `batch_summary.csv` (image_id, total, stained, unstained, % stained):

```bash
python -m cell_counter.cli batch --input-dir "Example Images" --out results/batch --save-overlay
```

Add `--auto-exemplars` to run the SAM3 backend on every pair with no manual boxing
(the interactive picker can't run headless, so auto‑exemplars is the way to use SAM3
in batch).

## Exit codes

`0` success · `2` bad input / unreadable image / bad config · `3` large alignment
residual (images look misaligned) · `4` `--require-sam3` set but SAM3 unavailable.
Non‑zero exits let you script batch loops safely.

## Project layout

```
cell_counter/
├── cli.py            # argparse entry point (run / calibrate / batch / pick-exemplars)
├── config.py         # Config dataclasses, YAML load, diameter rescaling
├── models.py         # CellRecord + per-stage result dataclasses
├── io_utils.py       # load/validate images, write json/csv
├── registration.py   # Stage 0b: alignment verify + optional warp
├── segmentation.py   # Stage 1: total-cell detection (filtered)
├── classification.py # Stage 2: stain classification (unfiltered)
├── metrics.py        # Stage 3: aggregate + QC
├── visualization.py  # Stage 4: overlay (numbered), debug panels, histograms
├── pipeline.py       # run_pipeline + calibration helpers (sweep, truth matching)
├── sam3_client.py    # torch-free client -> SAM3 subprocess bridge
├── sam3_bridge.py    # runs INSIDE the SAM3 venv (torch/transformers)
├── sam3_detection.py # SAM3 detection-union (recover missed cells)
├── sam3_auto.py      # auto-exemplars: bootstrap boxes from the color pass
├── sam3_pick.py      # torch-free driver for the exemplar picker
└── sam3_pick_gui.py  # interactive picker GUI (runs in the SAM3 venv)
configs/default.yaml  # all tunables, documented
tests/                # synthetic + real-crop regression tests (52, all passing)
```

## Tests

```bash
pip install pytest && pytest
```

The suite is torch‑free and hermetic: the SAM3 path is exercised by mocking the
subprocess boundary, and the real‑image regression pins c1's counts to their
validated bands. It runs without the SAM3 venv installed.
