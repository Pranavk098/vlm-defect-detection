"""
Evaluation utilities: accuracy, F1, confusion matrix on MVTec-AD validation split.

Provides the missing evaluation loop from the original project.
The original trained for 3 epochs and loss→0 with NO validation — classic overfit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _predict_batch(model, processor, images: list, question: str, cfg_inf) -> list[str]:
    from src.vlm_defect.data import build_conversation

    conversations = [build_conversation("") for _ in images]
    # Strip the assistant turn — we want the model to generate
    for c in conversations:
        c.pop()  # remove assistant entry

    texts = [
        processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in conversations
    ]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(
        model.device
    )

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg_inf.max_new_tokens,
            temperature=cfg_inf.temperature,
            do_sample=cfg_inf.do_sample,
        )

    # Decode only the newly generated tokens
    generated = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)


def _is_defect(text: str) -> bool:
    """Convert model output text to binary label."""
    t = text.strip().lower()
    return t.startswith("yes") or "anomaly" in t or "defect" in t


def evaluate(
    model,
    processor,
    json_path: str | Path,
    dataset_root: str | Path,
    cfg,
    batch_size: int = 4,
) -> dict:
    """
    Run inference on the val split and return metrics.

    Returns:
        dict with keys: accuracy, f1_defect, f1_normal, report, confusion_matrix
    """
    from src.vlm_defect.data import MVTecDataset

    val_ds = MVTecDataset(
        json_path=json_path,
        dataset_root=dataset_root,
        processor=processor,
        max_length=cfg.data.max_length,
        split="val",
        val_split=cfg.data.val_split,
    )

    y_true, y_pred = [], []

    for i in tqdm(range(0, len(val_ds), batch_size), desc="Evaluating"):
        batch = [val_ds[j] for j in range(i, min(i + batch_size, len(val_ds)))]
        images = [ex["image"] for ex in batch]
        gts = [ex["conversation"][-1]["content"][0]["text"] for ex in batch]

        preds = _predict_batch(model, processor, images, "", cfg.inference)

        for gt, pred in zip(gts, preds):
            y_true.append(int(_is_defect(gt)))
            y_pred.append(int(_is_defect(pred)))

    report = classification_report(y_true, y_pred, target_names=["normal", "defect"])
    cm = confusion_matrix(y_true, y_pred)

    logger.info("Evaluation report:\n%s", report)
    logger.info("Confusion matrix:\n%s", cm)

    correct = sum(a == b for a, b in zip(y_true, y_pred))
    return {
        "accuracy": correct / len(y_true),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }
