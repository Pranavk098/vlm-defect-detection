"""
Prepare MVTec-AD data → mvtec_train.json (LLaVA conversation format).

Improvements over original prepare_mvtec_json.py:
- Writes to data/ directory (keeps root clean)
- More descriptive defect answers ("Yes, I see a {defect} defect on the {category}.")
- Reproducible output (sorted glob)
- Validates that images exist before writing the JSON
- Prints per-category statistics

Usage:
    python scripts/prepare_data.py --root mvtec_anomaly_detection --out data/mvtec_train.json
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path


QUESTION = (
    "Is there any anomaly in this image? "
    "If yes, answer 'Yes' and describe the defect briefly. "
    "If no, just answer 'No'."
)

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def format_answer(defect_type: str, category: str) -> str:
    if defect_type == "good":
        return "No."
    readable = defect_type.replace("_", " ")
    return f"Yes, there is a {readable} defect on the {category}."


def build_record(image_rel: str, answer: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "image": image_rel,
        "conversations": [
            {"from": "human", "value": f"<image>\n{QUESTION}"},
            {"from": "gpt", "value": answer},
        ],
    }


def process_category(root: Path, category: str, records: list, stats: dict) -> None:
    cat_dir = root / category

    # Training split — only "good" samples
    train_good = cat_dir / "train" / "good"
    if train_good.is_dir():
        images = sorted(train_good.glob("*.png")) + sorted(train_good.glob("*.jpg"))
        for img in images:
            rel = img.relative_to(root).as_posix()
            records.append(build_record(rel, "No."))
        stats[category]["train_good"] = len(images)

    # Test split — good + all defect types
    test_dir = cat_dir / "test"
    if not test_dir.is_dir():
        return

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        images = sorted(defect_dir.glob("*.png")) + sorted(defect_dir.glob("*.jpg"))
        answer = format_answer(defect_type, category)
        for img in images:
            rel = img.relative_to(root).as_posix()
            records.append(build_record(rel, answer))
        stats[category][f"test_{defect_type}"] = len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MVTec-AD JSON for LLaVA training")
    parser.add_argument("--root", default="mvtec_anomaly_detection", help="MVTec dataset root")
    parser.add_argument("--out", default="data/mvtec_train.json", help="Output JSON path")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Download MVTec-AD from https://www.mvtec.com/company/research/datasets/mvtec-ad"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    stats: dict = {c: {} for c in MVTEC_CATEGORIES}

    for category in MVTEC_CATEGORIES:
        process_category(root, category, records, stats)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nDataset written to: {out_path}")
    print(f"Total records: {len(records):,}")
    print("\nPer-category breakdown:")
    for cat, s in stats.items():
        total = sum(s.values())
        if total > 0:
            print(f"  {cat:<15} {total:>5} images  {dict(s)}")


if __name__ == "__main__":
    main()
