"""Convert an MVTec ground-truth mask to a bounding box.

Masks live at <category>/ground_truth/<defect>/<name>_mask.png
(white = defective pixels). Output boxes use the 0-1000 normalized scale
shared by the grounding prompt (prompts/grounding_template.txt), so boxes
are resolution-independent.

Usage:
    python scripts/mask_to_bbox.py --mask <path-to-mask.png>
    python scripts/mask_to_bbox.py --mask-dir mvtec_anomaly_detection/bottle/ground_truth --output bboxes.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

SCALE = 1000


def mask_to_bbox(mask_path, scale=SCALE, threshold=127):
    """Return [x1, y1, x2, y2] in 0-`scale` coords, or None for an empty mask."""
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.nonzero(mask > threshold)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return [
        round(x1 / w * scale),
        round(y1 / h * scale),
        round(x2 / w * scale),
        round(y2 / h * scale),
    ]


def bbox_iou(a, b, scale=SCALE):
    """IoU of two [x1, y1, x2, y2] boxes on the same scale. Either may be None."""
    if not a or not b:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="MVTec mask -> bbox (0-1000 scale).")
    parser.add_argument("--mask", default=None, help="Single mask PNG.")
    parser.add_argument("--mask-dir", default=None, help="Recursively convert every *_mask.png below this dir.")
    parser.add_argument("--output", default=None, help="Write batch results as JSON {mask_path: bbox}.")
    args = parser.parse_args()

    if args.mask:
        print(json.dumps(mask_to_bbox(args.mask)))
    elif args.mask_dir:
        results = {}
        for path in sorted(Path(args.mask_dir).rglob("*_mask.png")):
            results[str(path)] = mask_to_bbox(str(path))
        n_empty = sum(1 for b in results.values() if b is None)
        print(f"Converted {len(results)} masks ({n_empty} empty).")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Saved to {args.output}")
    else:
        parser.error("Provide --mask or --mask-dir.")


if __name__ == "__main__":
    main()
