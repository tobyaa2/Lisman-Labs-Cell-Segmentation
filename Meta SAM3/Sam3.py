"""
SAM3 exemplar segmentation.

Give SAM3 one or more *example* boxes around an object in your image, and it
finds and segments every other instance of that same concept in the picture.
This is the "visual exemplar" prompt mode (as opposed to a text prompt).

Usage
-----
Interactive (left-drag exemplar boxes, or press 't' for a freehand lasso; scroll=zoom,
right-drag=pan, 'n'=toggle positive/negative, 'z'=undo, 'c'=clear, 'r'=reset, close to run):
    python Sam3.py --image path/to/your_image.jpg

Headless (pass the exemplar box explicitly, x1 y1 x2 y2 in pixels):
    python Sam3.py --image your_image.jpg --box 120 80 260 240

Multiple positive exemplars and negative exemplars:
    python Sam3.py --image img.jpg --box 120 80 260 240 --box 300 90 410 250 \
                   --neg-box 10 10 60 60

Options:
    --threshold   confidence cutoff for kept instances (default 0.3)
    --output      where to save the overlay (default: <image>_sam3.png)
    --model       HF checkpoint (default facebook/sam3)
"""

"""
cd "/Volumes/USBDevice/Lisman Labs/Meta SAM3"
.venv/bin/python Sam3.py --image /path/to/your_image.jpg
"""

import argparse
import os
import sys
from pathlib import Path

# Keep the downloaded MODEL on the SAME drive as this script (the USB drive),
# not the default ~/.cache on the internal disk. We override only HF_HUB_CACHE
# (where the multi-GB weights live) and deliberately leave HF_HOME at its
# default so the auth token in ~/.cache/huggingface/token is still found.
# Must be set BEFORE transformers/huggingface_hub are imported.
os.environ.setdefault("HF_HUB_CACHE", str(Path(__file__).resolve().parent / "hf_cache" / "hub"))

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser(description="SAM3 exemplar segmentation")
    p.add_argument("--image", required=True, help="Path to your image")
    p.add_argument(
        "--box",
        type=float,
        nargs=4,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Positive exemplar box in pixels (repeatable). Omit to draw it interactively.",
    )
    p.add_argument(
        "--neg-box",
        type=float,
        nargs=4,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Negative exemplar box: an example of what NOT to match (repeatable).",
    )
    p.add_argument("--threshold", type=float, default=0.3, help="Score threshold (default 0.3)")
    p.add_argument("--model", default="facebook/sam3", help="HF checkpoint")
    p.add_argument("--output", default=None, help="Output overlay path")
    return p.parse_args()


def post_process_memory_safe(outputs, target_size, threshold=0.3, mask_threshold=0.5, chunk=8):
    """Memory-safe re-implementation of Sam3's post_process_instance_segmentation.

    The stock processor upsamples EVERY kept mask to the full target size at once
    and casts to int64 (8 bytes/px), which OOMs on large images / many detections
    (e.g. dense microscopy fields). We instead: move everything to CPU (no MPS
    watermark cap), upsample masks in small chunks, and keep them as bool (1
    byte/px). Numerically identical: sigmoid-then-interpolate-then-threshold,
    same as the library.
    """
    import torch.nn.functional as F
    from transformers.models.sam3.image_processing_sam3 import _scale_boxes

    scores = outputs.pred_logits.sigmoid()                  # (B, Q)
    if outputs.presence_logits is not None:
        scores = scores * outputs.presence_logits.sigmoid()  # (B, 1) broadcast
    boxes = _scale_boxes(outputs.pred_boxes, [target_size])   # (B, Q, 4) xyxy in pixels

    scores, boxes = scores[0].cpu(), boxes[0].cpu()
    keep = scores > threshold
    scores, boxes = scores[keep], boxes[keep]
    mask_logits = outputs.pred_masks[0][keep].float().cpu()   # (K, h, w) small model-res

    mask_chunks = []
    for i in range(0, mask_logits.shape[0], chunk):
        m = mask_logits[i : i + chunk].sigmoid().unsqueeze(0)             # (1, c, h, w)
        m = F.interpolate(m, size=target_size, mode="bilinear", align_corners=False)
        mask_chunks.append((m.squeeze(0) > mask_threshold))               # bool (c, H, W)
    masks = (
        torch.cat(mask_chunks, dim=0)
        if mask_chunks
        else torch.zeros((0, *target_size), dtype=torch.bool)
    )
    # Sort by score, highest first, for stable/readable output.
    order = torch.argsort(scores, descending=True)
    return {"scores": scores[order], "boxes": boxes[order], "masks": masks[order]}


def pick_box_interactively(image: Image.Image):
    """Open a zoomable window; left-drag boxes or lasso freehand shapes around
    example objects.

    Draw as many exemplars as you like in one session, with either tool ('t'
    switches box <-> lasso). The lasso lets you trace irregular samples; since
    SAM3 exemplars are boxes, each trace is reduced to its tight bounding box
    for the model while the fluid outline stays on screen. Each shape is a
    positive example by default; press 'n' to toggle to negative examples
    (things to exclude). Returns (positive_boxes, negative_boxes), each a list
    of [x1, y1, x2, y2].

    Controls:
        left-drag    add an exemplar (box, or lasso polygon)
        t            switch drawing tool: box <-> lasso
        scroll       zoom in / out, centered on the cursor
        right-drag   pan
        n            toggle positive / negative for the NEXT shapes
        z / del      undo the last shape
        c            clear all shapes
        r            reset to the full view
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.widgets import RectangleSelector

    items = []          # list of dicts: {coords, label, patch, text}
    mode = {"neg": False}
    tool = {"name": "box"}   # active drawing tool: "box" or "lasso"
    full_xlim, full_ylim = (-0.5, image.width - 0.5), (image.height - 0.5, -0.5)  # y inverted for images
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(image)
    ax.set_xlim(full_xlim)
    ax.set_ylim(full_ylim)
    help_line = "left-drag=draw · t=box/lasso · scroll=zoom · right-drag=pan · n=+/- · z=undo · c=clear · r=reset · close=run"

    def refresh_title():
        pos = sum(1 for it in items if it["label"] == 1)
        neg = len(items) - pos
        adding = "NEGATIVE (exclude)" if mode["neg"] else "POSITIVE"
        ax.set_title(f"tool: {tool['name'].upper()}   |   adding: {adding}   |   "
                     f"{pos} positive, {neg} negative\n{help_line}")
        fig.canvas.draw_idle()

    refresh_title()

    def commit_item(coords, patch):
        """Add a finished exemplar. coords=[x1,y1,x2,y2] is the bounding box fed
        to SAM3; patch is the shape drawn on screen (a box, or a lasso polygon)."""
        label = 0 if mode["neg"] else 1
        color = "red" if mode["neg"] else "lime"
        patch.set(edgecolor=color, fill=False, linewidth=2)
        ax.add_patch(patch)
        text = ax.text(coords[0], coords[1] - 5, f"{'-' if label == 0 else '+'}{len(items) + 1}",
                       color=color, fontsize=11, fontweight="bold")
        items.append({"coords": coords, "label": label, "patch": patch, "text": text})
        refresh_title()

    def on_box(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return  # drag ended outside the axes
        coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            return  # zero-size (the selector's minspan already drops tiny strays)
        commit_item(coords, mpatches.Rectangle(
            (coords[0], coords[1]), coords[2] - coords[0], coords[3] - coords[1]))

    def on_lasso(verts):
        # SAM3 exemplars are boxes, so the freehand trace is reduced to its tight
        # bounding box for the model -- but tracing the object is quicker and gives
        # a snugger box than corner-dragging a rectangle. The fluid outline itself
        # is kept on screen so the selection still reads as the shape you drew.
        pts = [(x, y) for x, y in verts if x is not None and y is not None]
        if len(pts) < 3:
            return  # a click or stray, not a shape
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        coords = [min(xs), min(ys), max(xs), max(ys)]
        if coords[2] - coords[0] < 2 or coords[3] - coords[1] < 2:
            return  # too small to be a real trace
        commit_item(coords, mpatches.Polygon(pts, closed=True))

    # Box drawing uses ONE RectangleSelector on the LEFT button, exactly like the
    # original. The lasso is deliberately NOT a second widget: two _SelectorWidgets
    # on the same axes/button silently stop receiving drags on the macOS backend
    # (works on Agg, dies on 'macosx'). Instead the lasso is a manual freehand
    # handler on the same button (see on_press/on_motion/on_release), so there is
    # only ever ONE consumer of left-drag -- the arrangement the original proved
    # works on macOS. useblit=False: blitting caches a background that erases the
    # persistent shapes we add after each drag, so they'd never show on macOS; a
    # full redraw keeps committed shapes visible. interactive=False so the
    # rubber-band clears on release (we keep our own labeled patch instead).
    box_selector = RectangleSelector(
        ax, on_box, useblit=False, button=[1], interactive=False,
        minspanx=2, minspany=2, spancoords="pixels",
        props=dict(edgecolor="yellow", fill=False, linewidth=1.5, linestyle="--"),
    )
    # Live rubber-band for the lasso, drawn by the manual handlers below.
    lasso_line = Line2D([], [], color="yellow", linewidth=1.5, linestyle="--", visible=False)
    ax.add_line(lasso_line)
    lasso_state = {"pts": None}   # points collected while a freehand trace is in progress

    def cancel_lasso():
        if lasso_state["pts"] is not None:
            lasso_state["pts"] = None
            lasso_line.set_data([], [])
            lasso_line.set_visible(False)

    def set_tool(name):
        tool["name"] = name
        # Only the box tool is a real widget; the lasso lives in the manual
        # handlers, so all we toggle is whether the box selector listens.
        box_selector.set_active(name == "box")
        cancel_lasso()  # drop any half-drawn trace when switching tools
        refresh_title()

    def on_scroll(event):
        if event.inaxes is not ax:
            return
        factor = 0.8 if event.button == "up" else 1.25  # scroll up = zoom in
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xc, yc = event.xdata, event.ydata  # keep point under cursor fixed
        ax.set_xlim(xc + (x0 - xc) * factor, xc + (x1 - xc) * factor)
        ax.set_ylim(yc + (y0 - yc) * factor, yc + (y1 - yc) * factor)
        fig.canvas.draw_idle()

    pan = {}  # right-button drag pan, anchored to press-time limits (no drift)

    def on_press(event):
        if event.inaxes is not ax:
            return
        if event.button == 3:  # right-drag pan
            bbox = ax.get_window_extent()
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            pan["d"] = dict(
                sx=event.x, sy=event.y, xlim=(x0, x1), ylim=(y0, y1),
                xpp=(x1 - x0) / bbox.width, ypp=(y1 - y0) / bbox.height,
            )
        elif event.button == 1 and tool["name"] == "lasso" and event.xdata is not None:
            # start a freehand trace (box mode lets the RectangleSelector handle it)
            lasso_state["pts"] = [(event.xdata, event.ydata)]
            lasso_line.set_data([event.xdata], [event.ydata])
            lasso_line.set_visible(True)
            fig.canvas.draw_idle()

    def on_motion(event):
        d = pan.get("d")
        if d and event.x is not None:
            dx = (event.x - d["sx"]) * d["xpp"]
            dy = (event.y - d["sy"]) * d["ypp"]
            ax.set_xlim(d["xlim"][0] - dx, d["xlim"][1] - dx)
            ax.set_ylim(d["ylim"][0] - dy, d["ylim"][1] - dy)
            fig.canvas.draw_idle()
            return
        pts = lasso_state["pts"]
        if pts is not None and event.xdata is not None:  # extend the freehand trace
            pts.append((event.xdata, event.ydata))
            xs, ys = zip(*pts)
            lasso_line.set_data(xs, ys)
            fig.canvas.draw_idle()

    def on_release(event):
        pan.pop("d", None)
        pts = lasso_state["pts"]
        if pts is not None:  # finish the freehand trace -> reduce to its bounding box
            cancel_lasso()
            fig.canvas.draw_idle()
            on_lasso(pts)

    def remove_item(it):
        it["patch"].remove()
        it["text"].remove()

    def on_key(event):
        if event.key == "r":
            ax.set_xlim(full_xlim)
            ax.set_ylim(full_ylim)
            fig.canvas.draw_idle()
        elif event.key == "t":
            set_tool("lasso" if tool["name"] == "box" else "box")
        elif event.key == "n":
            mode["neg"] = not mode["neg"]
            refresh_title()
        elif event.key in ("z", "backspace", "delete") and items:
            remove_item(items.pop())
            refresh_title()
        elif event.key == "c":
            while items:
                remove_item(items.pop())
            refresh_title()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    set_tool("box")  # start in box mode (box selector active, lasso handlers idle)
    plt.show()

    pos = [it["coords"] for it in items if it["label"] == 1]
    neg = [it["coords"] for it in items if it["label"] == 0]
    if not pos:
        sys.exit("No positive exemplar was drawn. Re-run and left-drag at least one "
                 "box or lasso, or pass --box X1 Y1 X2 Y2.")
    return pos, neg


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image {image_path.name}  ({image.width}x{image.height})")

    # --- collect exemplar boxes ---
    if args.box:
        pos_boxes = args.box
        neg_boxes = args.neg_box or []
    else:
        pos_boxes, neg_boxes = pick_box_interactively(image)  # draw any number, +/- toggled with 'n'
    all_boxes = [list(map(float, b)) for b in pos_boxes + neg_boxes]
    labels = [1] * len(pos_boxes) + [0] * len(neg_boxes)  # 1 = positive, 0 = negative
    print(f"Exemplars: {len(pos_boxes)} positive, {len(neg_boxes)} negative")

    # --- load model (downloads from Hugging Face on first run) ---
    from transformers import Sam3Model, Sam3Processor

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model).to(device).eval()

    # input_boxes is nested per-image: [[box, box, ...]]; same for labels.
    inputs = processor(
        images=image,
        input_boxes=[all_boxes],
        input_boxes_labels=[labels],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    # Post-process on CPU in chunks to avoid MPS OOM on large images.
    results = post_process_memory_safe(
        outputs, target_size=(image.height, image.width), threshold=args.threshold
    )

    masks = results["masks"]    # (N, H, W) bool
    boxes = results["boxes"]    # (N, 4) xyxy
    scores = results["scores"]  # (N,)
    n = len(scores)
    print(f"\nFound {n} instance(s) at threshold {args.threshold}:")
    for i, s in enumerate(scores.tolist()):
        b = [round(v) for v in boxes[i].tolist()]
        print(f"  #{i + 1}  score={s:.3f}  box={b}")

    # --- render overlay ---
    overlay = image.convert("RGBA")
    rng = np.random.default_rng(0)
    for i in range(n):
        m = masks[i].cpu().numpy().astype(bool)
        color = tuple(int(c) for c in rng.integers(60, 256, size=3))
        tint = np.zeros((*m.shape, 4), dtype=np.uint8)
        tint[m] = (*color, 110)
        overlay = Image.alpha_composite(overlay, Image.fromarray(tint, "RGBA"))
    draw = ImageDraw.Draw(overlay)
    for i in range(n):
        x1, y1, x2, y2 = boxes[i].tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)
        draw.text((x1 + 3, y1 + 3), f"{scores[i]:.2f}", fill=(255, 255, 0, 255))
    # show the exemplar boxes you provided, in cyan/red
    for b, lab in zip(all_boxes, labels):
        draw.rectangle(b, outline=(0, 200, 255, 255) if lab else (255, 40, 40, 255), width=2)

    out_path = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_sam3.png")
    overlay.convert("RGB").save(out_path)
    print(f"\nSaved overlay -> {out_path}")


if __name__ == "__main__":
    main()
