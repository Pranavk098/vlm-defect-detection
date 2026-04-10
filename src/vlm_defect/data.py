"""MVTec AD data preparation and PyTorch Dataset for LLaVA fine-tuning.

Three public surfaces:
  1. create_dataset() / main()  — one-shot JSON builder (run via make prepare)
                                  writes mvtec_train.json (SFT) and mvtec_test.json
                                  (evaluation only — never touched during training)
  2. MVTecDataset + collate_fn  — used by the HuggingFace Trainer at training time
  3. load_split()               — returns (train_dataset, val_dataset) with a
                                  reproducible 10% held-out validation split

Improvement notes (v3):
  - Category-aware prompts: the category name (e.g. "screw", "capsule") is
    injected into the question so the model has explicit context about what
    it is inspecting.  This is especially helpful for visually similar categories
    where texture / shape alone is ambiguous.
  - Center-crop augmentation: for small-defect categories (screw, capsule,
    transistor, pill, metal_nut) a deterministic centre-crop is applied before
    the image is passed to the processor.  This effectively zooms into the
    centre 70% of the image, making fine-grained surface defects larger and
    therefore easier for the 336×336 vision encoder to detect.
  - apply_center_crop() is exported so evaluate.py can apply the same
    preprocessing during inference, preventing a train/eval discrepancy.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import WeightedRandomSampler

# ── Categories where defects are fine-grained / small relative to image area ─
# A 70% centre-crop is applied to these categories before the processor resizes
# to 336×336, effectively doubling the effective resolution for surface defects.
SMALL_DEFECT_CATEGORIES: frozenset[str] = frozenset(
    {"screw", "capsule", "transistor", "pill", "metal_nut"}
)

# Fraction of image width/height kept by the centre-crop for small-defect cats.
_CROP_FACTOR = 0.70


# ---------------------------------------------------------------------------
# Shared preprocessing helpers
# ---------------------------------------------------------------------------

def apply_center_crop(image: Image.Image, category: str) -> Image.Image:
    """Apply a deterministic centre-crop for small-defect categories.

    For categories in ``SMALL_DEFECT_CATEGORIES`` the centre 70% of the image
    is cropped and then resized back to the original dimensions.  This "zooms
    in" on the centre of the part, making surface-level scratches / dents
    larger in pixel space and therefore easier for the 336×336 vision encoder.

    For all other categories the image is returned unchanged.

    Args:
        image:    RGB PIL image to (optionally) crop.
        category: MVTec category name (e.g. "screw", "bottle").

    Returns:
        Cropped-and-resized PIL image for small-defect categories, original
        image unchanged for all others.
    """
    if category not in SMALL_DEFECT_CATEGORIES:
        return image
    w, h = image.size
    crop_w = int(w * _CROP_FACTOR)
    crop_h = int(h * _CROP_FACTOR)
    left   = (w - crop_w) // 2
    top    = (h - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.LANCZOS)


# ── Category-aware prompt template ───────────────────────────────────────────
# {category} is replaced at record-creation time with the actual category name
# (e.g. "bottle", "screw").  Giving the model explicit context about *what* it
# is inspecting reduces ambiguity for visually similar textures.
_QUESTION_TEMPLATE = (
    "<image>\n"
    "Is there any anomaly in this {category} image? "
    "If yes, say 'Yes, there is a <defect_type> anomaly.' "
    "If no, say 'No.'"
)


# ---------------------------------------------------------------------------
# JSON builder (make prepare / vlm-prepare)
# ---------------------------------------------------------------------------

def create_dataset(
    dataset_root: Path,
    output_train: Path,
    output_test: Path,
    anomaly_train_fraction: float = 0.5,
    seed: int = 42,
) -> None:
    """Walk *dataset_root* and write two LLaVA-format JSON files.

    Args:
        dataset_root: Root directory of the extracted MVTec AD dataset.
        output_train: Destination for train-split JSON.  Contains:
                      • All ``train/good/`` images  (normal, label "No.")
                      • ``anomaly_train_fraction`` of each defect type from
                        ``test/<defect>/`` (anomaly, labelled with defect name)
                        so the model learns the full response template.
        output_test:  Destination for test-split JSON.  Contains:
                      • All ``test/good/`` images
                      • The remaining (1 - anomaly_train_fraction) of each
                        defect type — never seen during training.
        anomaly_train_fraction: Fraction [0, 1] of test anomaly images to fold
                      into the training set.  Default 0.5.  Set to 0.0 to
                      revert to the original "good-only" SFT setup.
        seed:         RNG seed for the anomaly train/test split.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            "Download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad "
            "and extract to mvtec_anomaly_detection/"
        )

    rng = random.Random(seed)
    train_records: list[dict] = []
    test_records:  list[dict] = []

    for category_path in sorted(dataset_root.iterdir()):
        if not category_path.is_dir():
            continue

        category = category_path.name
        print(f"  Processing: {category}")

        # ── Train split: good images from train/ ─────────────────────────────
        train_good_path = category_path / "train" / "good"
        if train_good_path.exists():
            for img_path in sorted(train_good_path.glob("*.png")):
                relative = f"{category}/train/good/{img_path.name}"
                train_records.append(_make_record(relative, "No.", category))

        # ── Test split: good + defect images from test/ ───────────────────────
        test_path = category_path / "test"
        if not test_path.exists():
            continue

        for defect_dir in sorted(test_path.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_name = defect_dir.name
            is_anomaly  = defect_name != "good"
            imgs        = sorted(defect_dir.glob("*.png"))

            if not is_anomaly or anomaly_train_fraction == 0.0:
                # All good-test and all anomaly images (if fraction=0) go to test
                for img_path in imgs:
                    relative = f"{category}/test/{defect_name}/{img_path.name}"
                    answer   = f"Yes, there is a {defect_name} anomaly." if is_anomaly else "No."
                    test_records.append(_make_record(relative, answer, category))
            else:
                # Split anomaly images between train and test
                imgs_shuffled = imgs[:]
                rng.shuffle(imgs_shuffled)
                n_train = max(1, int(len(imgs_shuffled) * anomaly_train_fraction))
                train_imgs = imgs_shuffled[:n_train]
                test_imgs  = imgs_shuffled[n_train:]

                answer = f"Yes, there is a {defect_name} anomaly."
                for img_path in train_imgs:
                    relative = f"{category}/test/{defect_name}/{img_path.name}"
                    train_records.append(_make_record(relative, answer, category))
                for img_path in test_imgs:
                    relative = f"{category}/test/{defect_name}/{img_path.name}"
                    test_records.append(_make_record(relative, answer, category))

    # Shuffle train records so anomaly / normal examples are interleaved
    rng.shuffle(train_records)

    n_train_anomaly = sum(1 for r in train_records if r["conversations"][1]["value"] != "No.")
    n_train_normal  = len(train_records) - n_train_anomaly
    print(
        f"\nGenerated {len(train_records)} train samples "
        f"({n_train_normal} normal, {n_train_anomaly} anomaly), "
        f"{len(test_records)} test samples."
    )

    for path, records in ((output_train, train_records), (output_test, test_records)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved to {path}")


def _make_record(image_relative: str, answer: str, category: str) -> dict:
    """Return a single LLaVA conversation record with a category-aware prompt."""
    question = _QUESTION_TEMPLATE.format(category=category)
    return {
        "id": str(uuid.uuid4()),
        "image": image_relative,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt",   "value": answer},
        ],
    }


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
        "--output-train",
        type=Path,
        default=Path("data/mvtec_train.json"),
        help="Output JSON for the train split (default: data/mvtec_train.json)",
    )
    parser.add_argument(
        "--output-test",
        type=Path,
        default=Path("data/mvtec_test.json"),
        help="Output JSON for the test split (default: data/mvtec_test.json)",
    )
    parser.add_argument(
        "--anomaly-train-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction [0,1] of test anomaly images to include in training "
            "so the model learns defect-name templates (default: 0.5). "
            "Use 0.0 to revert to good-only SFT."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the anomaly train/test split (default: 42)",
    )
    args = parser.parse_args()
    create_dataset(
        args.dataset_root,
        args.output_train,
        args.output_test,
        anomaly_train_fraction=args.anomaly_train_fraction,
        seed=args.seed,
    )


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

    For categories in ``SMALL_DEFECT_CATEGORIES`` (screw, capsule, transistor,
    pill, metal_nut) a centre-crop is applied before passing the image to the
    processor, effectively zooming in on fine-grained surface defects.
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
        # Extract category from image path for centre-crop decision
        category = item["image"].split("/")[0]
        image = Image.open(self.image_folder / item["image"]).convert("RGB")
        # Apply zoom-in crop for small-defect categories
        image = apply_center_crop(image, category)

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

    n_anomaly = sum(1 for r in train_records if r["conversations"][1]["value"] != "No.")
    n_normal  = len(train_records) - n_anomaly
    print(
        f"[INFO] Split: {len(train_records)} train "
        f"({n_normal} normal / {n_anomaly} anomaly) "
        f"/ {len(val_records)} val"
    )

    train_ds = MVTecDataset(train_records, image_folder, processor, model_max_length)
    val_ds   = MVTecDataset(val_records,   image_folder, processor, model_max_length)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Weighted sampler — oversample hard / under-represented categories
# ---------------------------------------------------------------------------

def make_weighted_sampler(
    records: list[dict],
    category_weights: dict[str, float],
) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that oversamples hard categories.

    Each record's weight equals ``category_weights.get(category, 1.0)``.
    Categories not listed in *category_weights* keep weight 1.0.

    Example config (local_8gb.yaml)::

        category_weights:
          transistor: 4.0
          screw:      4.0
          capsule:    3.0
          pill:       2.5
          cable:      2.0

    Args:
        records: List of conversation records (same list passed to MVTecDataset).
        category_weights: Mapping of category name → sampling multiplier.

    Returns:
        A ``WeightedRandomSampler`` with ``len(records)`` samples and
        ``replacement=True`` (standard for weighted sampling).
    """
    weights: list[float] = []
    for record in records:
        # image path format: "<category>/train/good/<name>.png"
        category = record["image"].split("/")[0]
        weights.append(category_weights.get(category, 1.0))

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


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
