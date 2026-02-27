"""MVTec AD data preparation and PyTorch Dataset for LLaVA fine-tuning.

Three public surfaces:
  1. create_dataset() / main()  — one-shot JSON builder (run via make prepare)
  2. MVTecDataset + collate_fn  — used by the HuggingFace Trainer at training time
  3. load_split()               — returns (train_dataset, val_dataset) with a
                                  reproducible 10% held-out validation split
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# JSON builder (make prepare / vlm-prepare)
# ---------------------------------------------------------------------------

def create_dataset(dataset_root: Path, output_file: Path) -> None:
    """Walk *dataset_root* and write a LLaVA-format JSON to *output_file*."""
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
    import argparse

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


# ---------------------------------------------------------------------------
# PyTorch Dataset + collate — used by HuggingFace Trainer
# ---------------------------------------------------------------------------

class MVTecDataset(torch.utils.data.Dataset):
    """Load a list of MVTec conversation records and tokenise for LLaVA.

    Each item encodes the full USER→ASSISTANT conversation with the image.
    Prompt tokens are masked to -100 in ``labels`` so the model only learns
    to predict the assistant's reply.
    """

    def __init__(
        self,
        records: list[dict],
        image_folder: Path,
        processor,
        model_max_length: int = 2048,
    ) -> None:
        self.data = records
        self.image_folder = Path(image_folder)
        self.processor = processor
        self.model_max_length = model_max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        image = Image.open(self.image_folder / item["image"]).convert("RGB")

        human = item["conversations"][0]["value"]
        gpt = item["conversations"][1]["value"]

        # Full conversation fed to the model
        full_prompt = f"USER: {human} ASSISTANT: {gpt}</s>"
        # Prefix that must be masked in labels
        prefix_prompt = f"USER: {human} ASSISTANT: "

        enc = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
            max_length=self.model_max_length,
            truncation=True,
            padding=False,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values = enc["pixel_values"].squeeze(0)

        # Mask all prompt tokens; model learns only the assistant reply
        prefix_len = len(
            self.processor.tokenizer(
                prefix_prompt, add_special_tokens=False
            )["input_ids"]
        )
        labels = input_ids.clone()
        labels[:prefix_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def load_split(
    json_path: Path,
    image_folder: Path,
    processor,
    model_max_length: int = 2048,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[MVTecDataset, MVTecDataset]:
    """Load *json_path* and return a reproducible (train, val) dataset pair.

    Args:
        json_path: Path to the JSON file produced by ``create_dataset()``.
        image_folder: Root directory containing MVTec images.
        processor: HuggingFace processor (tokenizer + image processor).
        model_max_length: Maximum token length passed to the processor.
        val_fraction: Fraction of samples held out for validation (default 0.1).
        seed: Random seed for the shuffle so splits are reproducible.
    """
    with open(json_path) as f:
        records = json.load(f)

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]

    print(f"[INFO] Split: {len(train_records)} train / {len(val_records)} val")

    train_ds = MVTecDataset(train_records, image_folder, processor, model_max_length)
    val_ds = MVTecDataset(val_records, image_folder, processor, model_max_length)
    return train_ds, val_ds


def collate_fn(batch: list[dict]) -> dict:
    """Pad a batch of variable-length samples from MVTecDataset."""
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }
