"""Shared pytest fixtures.

Provides:
  fake_mvtec      — temp directory with a minimal MVTec-style layout
                    (2 train/good + 1 test/good + 1 test/broken_large images)
  fake_json_files — (train_path, test_path) JSON files pre-written by
                    create_dataset() for tests that need them on disk
  mock_processor  — stand-in for a HuggingFace LLaVA processor; returns
                    plausible dummy tensors without loading any weights
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from vlm_defect.data import create_dataset


# ---------------------------------------------------------------------------
# Fake MVTec directory
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_mvtec(tmp_path) -> Path:
    """Minimal MVTec-style dataset: one category ('bottle'), 4 tiny PNGs.

    Layout
    ------
    bottle/
      train/good/000.png, 001.png      ← 2 anomaly-free training images
      test/good/000.png                ← 1 anomaly-free test image
      test/broken_large/000.png        ← 1 defective test image
    """
    root = tmp_path / "mvtec"

    specs = [
        ("bottle/train/good", [(0, 0, 0), (80, 0, 0)]),
        ("bottle/test/good",  [(0, 128, 0)]),
        ("bottle/test/broken_large", [(255, 0, 0)]),
    ]
    for rel_dir, colors in specs:
        d = root / rel_dir
        d.mkdir(parents=True)
        for i, color in enumerate(colors):
            Image.new("RGB", (32, 32), color=color).save(d / f"{i:03d}.png")

    return root


# ---------------------------------------------------------------------------
# Pre-built JSON fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_json_files(fake_mvtec, tmp_path):
    """Run create_dataset() once and return (train_json_path, test_json_path)."""
    train_json = tmp_path / "data" / "mvtec_train.json"
    test_json  = tmp_path / "data" / "mvtec_test.json"
    create_dataset(fake_mvtec, train_json, test_json)
    return train_json, test_json


# ---------------------------------------------------------------------------
# Mock processor
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_processor():
    """HuggingFace processor stand-in that returns fixed dummy tensors.

    processor(text=..., images=..., return_tensors="pt")
        → {"input_ids": (1,10), "attention_mask": (1,10), "pixel_values": (1,3,32,32)}

    processor.tokenizer(text, add_special_tokens=False)
        → {"input_ids": [0, 1, 2, 3]}   (4-token prefix to mask in labels)
    """
    SEQ_LEN = 10
    proc = MagicMock()

    # Main __call__ — called by MVTecDataset.__getitem__
    proc.return_value = {
        "input_ids":      torch.zeros(1, SEQ_LEN, dtype=torch.long),
        "attention_mask": torch.ones(1,  SEQ_LEN, dtype=torch.long),
        "pixel_values":   torch.zeros(1, 3, 32, 32),
    }

    # tokenizer() — used to compute the prompt-prefix length for label masking
    proc.tokenizer.return_value = {"input_ids": [0, 1, 2, 3]}
    proc.tokenizer.padding_side = "right"

    return proc
