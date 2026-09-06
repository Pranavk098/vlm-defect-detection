"""Build a held-out eval split (eval_test.json) in LLaVA conversation format.

Two modes (like the chip project: a frozen JSON eval set, disjoint from train):

1. Stratified holdout (default) — sample `ratio` of mvtec_train.json per
   MVTec category with a fixed seed:
       python scripts/make_eval_split.py --input mvtec_train.json --output eval_test.json

2. Leakage-free rebuild — build eval directly from <root>/*/test/* (the
   MVTec-standard eval population). Recommended for final numbers because
   the legacy mvtec_train.json mixes train AND test images:
       python scripts/make_eval_split.py --from-mvtec-root mvtec_anomaly_detection --output eval_test.json

With --rewrite-train <path>, mode 1 additionally writes a decontaminated
train file with eval ids removed. Without it, training data is left
untouched and the id overlap is reported so you can exclude it manually.
"""

import argparse
import json
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES_15 = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


def category_of(item):
    return str(item.get("image", "")).split("/")[0]


def stratified_sample(items, ratio, seed):
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for item in items:
        by_cat[category_of(item)].append(item)
    eval_items, train_items = [], []
    for cat in sorted(by_cat):
        group = by_cat[cat]
        rng.shuffle(group)
        n_eval = max(1, round(len(group) * ratio))
        eval_items.extend(group[:n_eval])
        train_items.extend(group[n_eval:])
    rng.shuffle(eval_items)
    return train_items, eval_items


def build_from_mvtec_root(root):
    """Eval entries straight from <root>/<category>/test/<defect>/*.png."""
    root = Path(root)
    items = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name not in CATEGORIES_15:
            continue
        test_dir = cat_dir / "test"
        if not test_dir.exists():
            continue
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect = defect_dir.name
            is_anomaly = defect != "good"
            for img in sorted(defect_dir.glob("*.png")):
                if is_anomaly:
                    answer = f"Yes, there is a {defect} anomaly."
                else:
                    answer = "No."
                items.append({
                    "id": str(uuid.uuid4()),
                    "image": f"{cat_dir.name}/test/{defect}/{img.name}",
                    "conversations": [
                        {"from": "human", "value": "<image>\nIs there any anomaly in this image? Describe it."},
                        {"from": "gpt", "value": answer},
                    ],
                })
    return items


def main():
    parser = argparse.ArgumentParser(description="Build held-out eval_test.json.")
    parser.add_argument("--input", default="mvtec_train.json")
    parser.add_argument("--output", default="eval_test.json")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-mvtec-root", default=None)
    parser.add_argument("--rewrite-train", default=None,
                        help="Also write decontaminated train JSON to this path.")
    args = parser.parse_args()

    if args.from_mvtec_root:
        eval_items = build_from_mvtec_root(args.from_mvtec_root)
        print(f"Built eval from {args.from_mvtec_root}: {len(eval_items)} items (all of */test/*).")
    else:
        with open(args.input, encoding="utf-8") as f:
            items = json.load(f)
        _, eval_items = stratified_sample(items, args.ratio, args.seed)
        print(f"Held out {len(eval_items)}/{len(items)} items (ratio={args.ratio}, seed={args.seed}).")
        if args.rewrite_train:
            eval_ids = {e["id"] for e in eval_items}
            train_items = [x for x in items if x["id"] not in eval_ids]
            with open(args.rewrite_train, "w", encoding="utf-8") as f:
                json.dump(train_items, f, indent=2)
            print(f"Wrote decontaminated train ({len(train_items)} items) to {args.rewrite_train}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_items, f, indent=2)
    counts = Counter(category_of(e) for e in eval_items)
    print(f"Saved {len(eval_items)} eval items to {args.output}")
    for cat in sorted(counts):
        print(f"  {cat:12s} {counts[cat]}")


if __name__ == "__main__":
    main()
