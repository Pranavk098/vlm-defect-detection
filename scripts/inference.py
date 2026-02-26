"""
Run inference on one or more images with the fine-tuned checkpoint.

Usage:
    # Single image
    python scripts/inference.py --checkpoint checkpoints/llava-mvtec-lora \
        --image path/to/image.png

    # Directory of images
    python scripts/inference.py --checkpoint checkpoints/llava-mvtec-lora \
        --image_dir mvtec_anomaly_detection/bottle/test

    # Evaluate on val split
    python scripts/inference.py --checkpoint checkpoints/llava-mvtec-lora \
        --evaluate --config configs/local_8gb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm_defect.config import load_config
from src.vlm_defect.model import load_inference_model


QUESTION = (
    "Is there any anomaly in this image? "
    "If yes, answer 'Yes' and describe the defect briefly. "
    "If no, just answer 'No'."
)


def predict_single(model, processor, image: Image.Image, cfg) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.inference.max_new_tokens,
            temperature=cfg.inference.temperature,
            do_sample=cfg.inference.do_sample,
        )

    generated = out[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to saved LoRA checkpoint dir")
    parser.add_argument("--config", default="configs/local_8gb.yaml")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--image_dir", help="Directory of images")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on val split and print metrics",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, processor = load_inference_model(args.checkpoint, cfg)

    if args.evaluate:
        from src.vlm_defect.evaluate import evaluate

        results = evaluate(
            model,
            processor,
            json_path=cfg.data.json_path,
            dataset_root=cfg.data.dataset_root,
            cfg=cfg,
        )
        print(f"\nAccuracy: {results['accuracy']:.4f}")
        print(results["report"])
        return

    image_paths: list[Path] = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.image_dir:
        d = Path(args.image_dir)
        image_paths.extend(sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")))

    if not image_paths:
        parser.error("Provide --image, --image_dir, or --evaluate")

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        answer = predict_single(model, processor, image, cfg)
        print(f"{img_path.name}: {answer}")


if __name__ == "__main__":
    main()
