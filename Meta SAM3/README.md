# SAM3 Exemplar Segmentation

Draw one or more **example** shapes around an object in an image, and
[Meta's SAM3](https://huggingface.co/facebook/sam3) finds and segments **every
other instance of that same concept** in the picture. This is SAM3's *visual
exemplar* prompt mode (as opposed to a text prompt) — point at one cell, get all
the cells.

Everything lives in a single script: [`Sam3.py`](Sam3.py).

<!-- Example overlays live in Images/*_sam3.png -->

---

## What it does

- **Interactive picker** — a zoomable window where you draw exemplars with a
  **box** or a freehand **lasso**, mark things to *exclude*, then close the
  window to run.
- **Headless mode** — pass exemplar boxes on the command line, no GUI.
- Outputs a PNG overlay with each detected instance tinted a random color, plus
  the detection boxes and your exemplar boxes.

The heavy model weights are **not** in this repository (see
[Installation](#installation)); they download from Hugging Face on first run.

---

## Requirements

- **Python 3.11+** (developed on 3.13)
- **~4 GB free disk** for the model weights (~3.2 GB) + cache
- A GPU is optional — the script auto-selects **Apple MPS**, **CUDA**, or **CPU**
- A **Hugging Face account with access to the gated `facebook/sam3` repo**

---

## Installation

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd "Meta SAM3"
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch note.** `requirements.txt` installs the default CPU/MPS build of
> PyTorch. For an NVIDIA CUDA build, install `torch` from
> [pytorch.org](https://pytorch.org) *first* (with the right CUDA index URL),
> then run the `pip install -r requirements.txt` above.
>
> **transformers note.** SAM3 support is only in the `main` branch of
> transformers at the time of writing, so `requirements.txt` installs it from
> source. Tested with `transformers==5.13.0.dev0`.

### 3. Get access to the model (it's gated)

The `facebook/sam3` repository requires manual approval:

1. Sign in at [huggingface.co](https://huggingface.co).
2. Open **https://huggingface.co/facebook/sam3** and click **“Request access”**.
   Approval is usually quick.
3. Log in locally so the download is authorized (this saves a token to
   `~/.cache/huggingface/token`):

   ```bash
   hf auth login          # older clients: huggingface-cli login
   ```

### 4. First run downloads the weights

The model is **not** in git — that's why you did step 3. The **first** time you
run `Sam3.py`, it downloads `facebook/sam3` (~3.2 GB) into a local
**`hf_cache/`** folder beside the script, and reuses it on every later run. That
folder is in [`.gitignore`](.gitignore) and must never be committed.

```bash
python Sam3.py --image Images/your_image.jpg
```

That's it. Every subsequent run is offline-fast.

---

## Usage

### Interactive (recommended)

```bash
python Sam3.py --image path/to/your_image.jpg
```

A window opens. Draw one or more example shapes around a target object, then
**close the window to run**. The overlay is saved next to your image as
`<image>_sam3.png`.

**Controls**

| Action | Control |
|---|---|
| Draw an exemplar | **left-drag** (a box, or a lasso outline) |
| Switch tool: box ⇄ lasso | **`t`** |
| Zoom in / out (at cursor) | **scroll wheel** |
| Pan | **right-drag** |
| Toggle positive / negative for the next shapes | **`n`** |
| Undo last shape | **`z`** / `delete` / `backspace` |
| Clear all shapes | **`c`** |
| Reset to full view | **`r`** |
| Run segmentation | **close the window** |

- **Positive** exemplars (green) say *“find things like this.”*
- **Negative** exemplars (red, press `n`) say *“…but not things like this.”*
- The **lasso** lets you trace irregular samples. SAM3 exemplars are boxes, so a
  trace is reduced to its tight bounding box for the model — tracing is just a
  faster, snugger way to place that box than corner-dragging a rectangle.

### Headless (no GUI)

Pass exemplar boxes explicitly as pixel coordinates `x1 y1 x2 y2`:

```bash
python Sam3.py --image your_image.jpg --box 120 80 260 240
```

Multiple positive exemplars and a negative exemplar:

```bash
python Sam3.py --image img.jpg \
  --box 120 80 260 240 --box 300 90 410 250 \
  --neg-box 10 10 60 60
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--image` | *(required)* | Path to the input image |
| `--box X1 Y1 X2 Y2` | — | Positive exemplar box, in pixels. Repeatable. Omit to draw interactively. |
| `--neg-box X1 Y1 X2 Y2` | — | Negative exemplar (an example of what *not* to match). Repeatable. |
| `--threshold` | `0.3` | Confidence cutoff for kept instances. Lower = more detections. |
| `--model` | `facebook/sam3` | Hugging Face checkpoint |
| `--output` | `<image>_sam3.png` | Where to save the overlay |

---

## Output

- The console prints how many instances were found and each one's score + box.
- An overlay PNG (`<image>_sam3.png` by default) shows every detected instance
  as a translucent colored mask, green detection boxes with scores, and your
  cyan/red exemplar boxes.

Tune `--threshold` if you get too many or too few detections.

---

## Troubleshooting

**`UnicodeDecodeError` on `import transformers` (macOS + USB/exFAT drive).**
exFAT drives sprout thousands of AppleDouble `._*` metadata files inside the
venv; the transformers import scanner chokes on them. Delete them before running:

```bash
find .venv -name '._*' -delete
```

**`GatedRepoError` / HTTP 401 when downloading the model.** You either haven't
been granted access to `facebook/sam3` (do step 3 above) or aren't logged in.
Verify with `hf auth whoami`. Do **not** set the `HF_HOME` environment variable
to relocate the cache — it moves the token lookup and breaks auth. `Sam3.py`
deliberately only sets `HF_HUB_CACHE` (to `./hf_cache/hub`) so the token at
`~/.cache/huggingface/token` is still found.

**Out of memory during post-processing on large / dense images.** Already
handled: `Sam3.py` ships `post_process_memory_safe()`, which post-processes on
CPU and upsamples masks in chunks instead of all at once. (A 2893×2545 image with
126 detections would otherwise blow past Apple MPS memory.)

**`404` for the model.** The repo is `facebook/sam3` (not `facebook/sam3-base`,
which some docstrings show). There is also a `facebook/sam3.1`.

**No window appears / matplotlib backend error.** Make sure you're running in an
environment with a display and a GUI backend installed (the default on macOS and
most desktop Python installs). Headless servers should use the `--box` mode.

---

## Repository layout

```
Sam3.py            # the whole program
requirements.txt   # Python dependencies
README.md          # this file
Images/            # your input images (and generated *_sam3.png overlays)
hf_cache/          # downloaded model weights — created on first run, NOT in git
.venv/             # your virtual environment — NOT in git
```

The model in `hf_cache/` and the `.venv/` are intentionally excluded by
[`.gitignore`](.gitignore); anyone cloning the repo recreates them via the
[Installation](#installation) steps.
