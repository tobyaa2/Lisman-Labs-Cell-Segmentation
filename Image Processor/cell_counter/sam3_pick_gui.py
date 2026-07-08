"""Interactive exemplar picker GUI — runs INSIDE the Meta SAM3 venv (needs
matplotlib). Reuses Sam3.py's `pick_box_interactively`.

    <sam3_venv_python> sam3_pick_gui.py <image.png> <out.json> <prompt> <meta_dir>

Writes {"pos": [[x1,y1,x2,y2],...], "neg": [...]} (pixel xyxy) to <out.json>.
"""
import json
import sys


def main():
    image_path, out_path, prompt, meta_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    sys.path.insert(0, meta_dir)
    from PIL import Image
    from Sam3 import pick_box_interactively

    if prompt:
        print(prompt)
    image = Image.open(image_path).convert("RGB")
    pos, neg = pick_box_interactively(image)
    with open(out_path, "w") as fh:
        json.dump({"pos": [list(map(float, b)) for b in pos],
                   "neg": [list(map(float, b)) for b in neg]}, fh)


if __name__ == "__main__":
    main()
