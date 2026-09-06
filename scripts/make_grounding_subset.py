"""Build a 200-image grounding subset (mvtec_grounding_200.json).

For defective MVTec test images that have a ground-truth mask, derive a
bbox via scripts/mask_to_bbox.py and pair it with the grounding prompt in
prompts/grounding_template.txt:

    human: <image> + "Locate the defect as [x1, y1, x2, y2] (0-1000 scale) and explain in one sentence."
    gpt:   "Defect at [x1, y1, x2, y2]. <one-sentence rationale naming the defect type>."

Sampling is stratified across the 15 categories (~13-14 each, seed 42);
shortfall categories (e.g. toothbrush) are topped up from larger ones and
every shortfall is reported — the output always states its real per-class
counts, never a silently imbalanced 200.

Usage:
    python scripts/make_grounding_subset.py --root mvtec_anomaly_detection --output mvtec_grounding_200.json
"""

import argparse
import json
import random
import uuid
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mask_to_bbox import mask_to_bbox

CATEGORIES_15 = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]

DEFAULT_HUMAN = ("<image>\nLocate the defect as [x1, y1, x2, y2] "
                 "(coordinates in 0-1000 scale) and explain in one sentence.")


def load_template(path):
    try:
        return (Path(path).read_text(encoding="utf-8").strip()
                .replace("{IMAGE_TOKEN}", "<image>"))
    except OSError:
        return DEFAULT_HUMAN


def collect_candidates(root):
    """All (category, defect, image, mask) tuples with a non-empty mask bbox."""
    root = Path(root)
    by_cat = {c: [] for c in CATEGORIES_15}
    for cat in CATEGORIES_15:
        test_dir = root / cat / "test"
        gt_dir = root / cat / "ground_truth"
        if not test_dir.exists() or not gt_dir.exists():
            continue
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir() or defect_dir.name == "good":
                continue
            for img in sorted(defect_dir.glob("*.png")):
                mask = gt_dir / defect_dir.name / f"{img.stem}_mask.png"
                if not mask.exists():
                    continue
                bbox = mask_to_bbox(str(mask))
                if bbox is None:  # empty mask -> no localizable defect
                    continue
                by_cat[cat].append((defect_dir.name, img, mask, bbox))
    return by_cat


def main():
    parser = argparse.ArgumentParser(description="Build 200-image grounding subset.")
    parser.add_argument("--root", default="mvtec_anomaly_detection")
    parser.add_argument("--output", default="mvtec_grounding_200.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template", default="prompts/grounding_template.txt")
    args = parser.parse_args()

    human_prompt = load_template(args.template)
    by_cat = collect_candidates(args.root)
    avail = {c: len(v) for c, v in by_cat.items()}
    print("Candidates with non-empty masks per category:")
    for c in CATEGORIES_15:
        print(f"  {c:12s} {avail[c]}")

    rng = random.Random(args.seed)
    quota, remainder = divmod(args.n, len(CATEGORIES_15))
    picked = []
    pool = []
    for i, cat in enumerate(CATEGORIES_15):
        cands = by_cat[cat][:]
        rng.shuffle(cands)
        take = quota + (1 if i < remainder else 0)
        picked.extend([(cat,) + c for c in cands[:take]])
        pool.extend([(cat,) + c for c in cands[take:]])
    if len(picked) < args.n:  # top up shortfall categories from the pool
        rng.shuffle(pool)
        picked.extend(pool[:args.n - len(picked)])
        print(f"Topped up {args.n - len(picked) + len(pool[:args.n - len(picked)])} items from larger categories.")
    rng.shuffle(picked)
    picked = picked[:args.n]

    items = []
    for cat, defect, img, mask, bbox in picked:
        rationale = f"There is a {defect} defect at {bbox}."
        items.append({
            "id": str(uuid.uuid4()),
            "image": f"{cat}/test/{defect}/{img.name}",
            "mask": f"{cat}/ground_truth/{defect}/{mask.name}",
            "category": cat,
            "defect": defect,
            "bbox_1000": bbox,
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": f"Defect at {bbox}. {rationale}"},
            ],
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    counts = Counter(e["category"] for e in items)
    print(f"Saved {len(items)} grounding items to {args.output}")
    for c in sorted(counts):
        flag = "" if avail[c] >= quota else "  (shortfall category)"
        print(f"  {c:12s} {counts[c]}{flag}")


if __name__ == "__main__":
    main()
