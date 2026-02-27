"""Evaluation script — accuracy, F1, and confusion matrix on the val split.

Usage:
    python scripts/evaluate.py checkpoints/llava-mvtec-lora configs/local_8gb.yaml
    make eval CHECKPOINT=checkpoints/llava-mvtec-lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


def _is_anomaly_response(text: str) -> bool:
    """Return True if the model response indicates an anomaly ('Yes')."""
    return text.strip().lower().startswith("yes")


def evaluate(checkpoint_dir: Path, cfg: dict, project_dir: Path) -> dict:
    """Run inference over the val split and return metrics.

    Returns a dict with keys: accuracy, precision, recall, f1,
    confusion_matrix (as {tp, fp, tn, fn}).
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel

    from vlm_defect.data import load_split

    data_json = project_dir / cfg["data"]["path"]
    image_folder = project_dir / cfg["data"]["image_folder"]
    val_fraction = cfg["data"].get("val_fraction", 0.1)

    print("[INFO] Loading processor...")
    base_model_id = cfg["model"]["name_or_path"]
    processor = AutoProcessor.from_pretrained(base_model_id)

    print("[INFO] Loading base model + LoRA adapter...")
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model.eval()

    print("[INFO] Loading val split...")
    _, val_ds = load_split(
        json_path=data_json,
        image_folder=image_folder,
        processor=processor,
        model_max_length=cfg["training"]["model_max_length"],
        val_fraction=val_fraction,
    )

    tp = fp = tn = fn = 0

    with torch.inference_mode():
        for idx in range(len(val_ds)):
            item = val_ds.data[idx]
            image = Image.open(image_folder / item["image"]).convert("RGB")
            human = item["conversations"][0]["value"]
            gpt_truth = item["conversations"][1]["value"]

            prompt = f"USER: {human} ASSISTANT:"
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )
            generated = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            pred_anomaly = _is_anomaly_response(generated)
            true_anomaly = _is_anomaly_response(gpt_truth)

            if true_anomaly and pred_anomaly:
                tp += 1
            elif not true_anomaly and pred_anomaly:
                fp += 1
            elif true_anomaly and not pred_anomaly:
                fn += 1
            else:
                tn += 1

            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(val_ds)} evaluated...")

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_samples": total,
    }
    return metrics


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned LLaVA checkpoint on the MVTec val split"
    )
    parser.add_argument("checkpoint", type=Path, help="Path to LoRA checkpoint dir")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_dir = Path(__file__).parent.parent.parent.absolute()

    if not (project_dir / cfg["data"]["path"]).exists():
        print(f"[ERROR] Training data not found. Run: make prepare")
        sys.exit(1)

    metrics = evaluate(args.checkpoint, cfg, project_dir)

    print("\n── Evaluation Results ──────────────────────")
    print(f"  Samples:   {metrics['n_samples']}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion: TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")

    out_path = args.checkpoint / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
