"""
Dataset and data collator for LLaVA fine-tuning on MVTec-AD.

Key design choices:
- Uses HuggingFace-native LlavaProcessor (no custom tokenizer hacks)
- Builds a proper train/val split (prevents the loss→0 overfitting seen in original)
- DataCollator handles image+text batching correctly for SFTTrainer
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert quality-control inspector. "
    "Examine the image and answer whether there is an anomaly."
)

QUESTION = "Is there any anomaly in this image? If yes, describe it briefly."


def build_conversation(answer: str) -> list[dict]:
    """Return a standard LLaVA-format conversation list."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MVTecDataset(Dataset):
    """
    Wraps mvtec_train.json and resolves image paths.

    The JSON was produced by prepare_data.py and has the schema:
        [{"id": str, "image": str, "conversations": [...]}]

    The 'image' field is a relative path like 'bottle/train/good/000.png'.
    """

    def __init__(
        self,
        json_path: str | Path,
        dataset_root: str | Path,
        processor,
        max_length: int = 512,
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
    ):
        self.dataset_root = Path(dataset_root)
        self.processor = processor
        self.max_length = max_length

        with open(json_path) as f:
            records = json.load(f)

        # Reproducible shuffle + split
        rng = random.Random(seed)
        rng.shuffle(records)
        n_val = max(1, int(len(records) * val_split))

        if split == "train":
            self.records = records[n_val:]
        elif split == "val":
            self.records = records[:n_val]
        else:
            self.records = records  # full dataset (inference)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        img_path = self.dataset_root / rec["image"]

        image = Image.open(img_path).convert("RGB")

        # Pull the assistant answer from the stored conversation
        answer = rec["conversations"][1]["value"]
        conversation = build_conversation(answer)

        return {
            "image": image,
            "conversation": conversation,
            "id": rec["id"],
        }


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class LLaVACollator:
    """
    Collates image + conversation samples into model-ready tensors.

    Handles:
    - Applying the processor's chat template
    - Padding to the longest sequence in the batch
    - Setting labels = -100 for padding + image tokens (we only compute loss
      on the assistant's text tokens)
    """

    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        texts, images = [], []

        for ex in examples:
            images.append(ex["image"])
            text = self.processor.apply_chat_template(
                ex["conversation"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Build labels: mask everything except the assistant response tokens
        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Mask image placeholder tokens (processor inserts -200 or similar)
        # LLaVA uses IMAGE_TOKEN_INDEX = -200 internally; the processor
        # represents them as a specific token id — mask them in labels.
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_token
        )
        labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_datasets(cfg_data, processor) -> tuple[MVTecDataset, MVTecDataset]:
    """Return (train_dataset, val_dataset) from a DataConfig."""
    common = dict(
        json_path=cfg_data.json_path,
        dataset_root=cfg_data.dataset_root,
        processor=processor,
        max_length=cfg_data.max_length,
        val_split=cfg_data.val_split,
    )
    train_ds = MVTecDataset(**common, split="train")
    val_ds = MVTecDataset(**common, split="val")
    return train_ds, val_ds
