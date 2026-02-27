#!/usr/bin/env python3
"""Prepare MVTec AD dataset in LLaVA conversation format.

Usage:
    python scripts/prepare_data.py                            # defaults
    python scripts/prepare_data.py --dataset-root /data/mvtec --output data/mvtec_train.json
    make prepare                                              # via Makefile
"""

import argparse
import json
import uuid
from pathlib import Path


def create_dataset(dataset_root: Path, output_file: Path) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            "Download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad "
            "and extract to mvtec_anomaly_detection/"
        )

    dataset = []

    for category_path in sorted(dataset_root.iterdir()):
        if not category_path.is_dir():
            continue

        print(f"  Processing: {category_path.name}")

        # Training split — only 'good' samples
        train_path = category_path / "train" / "good"
        if train_path.exists():
            for img_path in sorted(train_path.glob("*.png")):
                relative = f"{category_path.name}/train/good/{img_path.name}"
                dataset.append({
                    "id": str(uuid.uuid4()),
                    "image": relative,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nIs there any anomaly in this image? Answer 'Yes' or 'No'.",
                        },
                        {"from": "gpt", "value": "No."},
                    ],
                })

        # Test split — good + defect sub-folders
        test_path = category_path / "test"
        if test_path.exists():
            for defect_type_path in sorted(test_path.iterdir()):
                if not defect_type_path.is_dir():
                    continue

                defect_name = defect_type_path.name
                is_anomaly = defect_name != "good"

                for img_path in sorted(defect_type_path.glob("*.png")):
                    relative = f"{category_path.name}/test/{defect_name}/{img_path.name}"
                    answer = (
                        f"Yes, there is a {defect_name} anomaly."
                        if is_anomaly
                        else "No."
                    )
                    dataset.append({
                        "id": str(uuid.uuid4()),
                        "image": relative,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>\nIs there any anomaly in this image? Describe it.",
                            },
                            {"from": "gpt", "value": answer},
                        ],
                    })

    print(f"\nGenerated {len(dataset)} samples.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MVTec AD dataset for LLaVA fine-tuning"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("mvtec_anomaly_detection"),
        help="Root directory of the extracted MVTec AD dataset (default: mvtec_anomaly_detection/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mvtec_train.json"),
        help="Output JSON file path (default: data/mvtec_train.json)",
    )
    args = parser.parse_args()
    create_dataset(args.dataset_root, args.output)


if __name__ == "__main__":
    main()
