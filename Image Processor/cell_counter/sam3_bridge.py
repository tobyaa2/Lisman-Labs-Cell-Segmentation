"""SAM3 inference bridge — runs INSIDE the Meta SAM3 venv, never imported by the
package.

`cell_counter` (torch-free) invokes this as a subprocess:

    <sam3_venv_python> sam3_bridge.py <request.json>

`request.json` describes one model load and one or more exemplar-segmentation
"passes" (a concept = a set of pixel-xyxy exemplar boxes + labels on one image).
For each pass we write `<out>.npz` with bit-packed instance masks + scores + boxes,
and print a JSON manifest to stdout. See `sam3_client.py` for the other side.
"""
import json
import os
import sys


def _read_request(path):
    with open(path) as fh:
        return json.load(fh)


def post_process_memory_safe(outputs, target_size, threshold=0.3, mask_threshold=0.5, chunk=8):
    """Vendored from Meta SAM3's Sam3.py: memory-safe instance post-processing.
    Move to CPU, upsample masks in small chunks, keep bool. Numerically identical
    to the library's post_process_instance_segmentation."""
    import torch
    import torch.nn.functional as F
    from transformers.models.sam3.image_processing_sam3 import _scale_boxes

    scores = outputs.pred_logits.sigmoid()
    if outputs.presence_logits is not None:
        scores = scores * outputs.presence_logits.sigmoid()
    boxes = _scale_boxes(outputs.pred_boxes, [target_size])
    scores, boxes = scores[0].cpu(), boxes[0].cpu()
    keep = scores > threshold
    scores, boxes = scores[keep], boxes[keep]
    mask_logits = outputs.pred_masks[0][keep].float().cpu()
    mask_chunks = []
    for i in range(0, mask_logits.shape[0], chunk):
        m = mask_logits[i:i + chunk].sigmoid().unsqueeze(0)
        m = F.interpolate(m, size=target_size, mode="bilinear", align_corners=False)
        mask_chunks.append((m.squeeze(0) > mask_threshold))
    masks = (torch.cat(mask_chunks, dim=0) if mask_chunks
             else torch.zeros((0, *target_size), dtype=torch.bool))
    order = torch.argsort(scores, descending=True)
    return {"scores": scores[order], "boxes": boxes[order], "masks": masks[order]}


def main():
    req = _read_request(sys.argv[1])

    # Point HF at the local cache BEFORE importing transformers (mirrors Sam3.py).
    hub = req.get("hf_hub_cache") or os.path.join(req["meta_dir"], "hf_cache", "hub")
    os.environ.setdefault("HF_HUB_CACHE", hub)

    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model_id = req.get("model_id", "facebook/sam3")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device).eval()

    score_thr = req.get("score_threshold", 0.3)
    mask_thr = req.get("mask_threshold", 0.5)
    manifest = {"device": device, "model_id": model_id, "passes": []}

    for p in req["passes"]:
        img = Image.open(p["image"]).convert("RGB")
        boxes = [[float(v) for v in b] for b in p["boxes"]]
        labels = [int(v) for v in p["labels"]]
        inputs = processor(images=img, input_boxes=[boxes],
                           input_boxes_labels=[labels], return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        res = post_process_memory_safe(outputs, target_size=(img.height, img.width),
                                       threshold=score_thr, mask_threshold=mask_thr)
        masks = res["masks"].cpu().numpy().astype(bool)      # (K, H, W)
        scores = res["scores"].cpu().numpy().astype("float32")
        boxes_out = res["boxes"].cpu().numpy().astype("float32")
        H, W = img.height, img.width
        # bit-pack masks (8x smaller) so dense 8MP fields stay compact on disk
        packed = np.packbits(masks.reshape(len(masks), -1), axis=1) if len(masks) else \
            np.zeros((0, 0), dtype=np.uint8)
        np.savez_compressed(p["out"], masks_packed=packed, n=len(masks),
                            shape=np.array([len(masks), H, W]), scores=scores, boxes=boxes_out)
        manifest["passes"].append({"concept": p["concept"], "n": int(len(masks)),
                                   "out": p["out"], "H": H, "W": W})

    sys.stdout.write(json.dumps(manifest))


if __name__ == "__main__":
    main()
