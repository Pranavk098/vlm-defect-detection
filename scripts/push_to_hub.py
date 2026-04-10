#!/usr/bin/env python3
"""Merge LoRA adapter into base model and push to HuggingFace Hub.

Steps performed:
  1. Load base model in bfloat16 full precision (NOT 4-bit) — required for a
     mathematically correct merge.  Merging a quantised model produces degraded
     weights because the NF4 quantisation error is permanently baked in.
  2. Load LoRA adapter via PeftModel.from_pretrained().
  3. merge_and_unload() — folds adapter weights into base; no PEFT dependency
     needed at inference time after this step.
  4. Push merged model, processor, and MODEL_CARD.md to the Hub.

Memory note:
  Loading LLaVA-1.5-7B in bfloat16 requires ~14 GB of RAM/VRAM.
  If your GPU has < 16 GB VRAM, use --device cpu to load on CPU RAM instead
  (slower, but correct).  A 32 GB RAM machine can handle this comfortably.

Usage:
    python scripts/push_to_hub.py \\
        --checkpoint checkpoints/llava-mvtec-lora-v3/checkpoint-500 \\
        --repo-id your-username/llava-mvtec-defect-detection \\
        --config configs/local_8gb.yaml

    # Load on CPU (if GPU VRAM < 16 GB):
    python scripts/push_to_hub.py \\
        --checkpoint checkpoints/llava-mvtec-lora-v3/checkpoint-500 \\
        --repo-id your-username/llava-mvtec-defect-detection \\
        --config configs/local_8gb.yaml \\
        --device cpu

    # Private repo:
    python scripts/push_to_hub.py ... --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base model and push to HuggingFace Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/llava-mvtec-lora-v3/checkpoint-500"),
        help="Path to the LoRA checkpoint directory "
             "(default: checkpoints/llava-mvtec-lora-v3/checkpoint-500)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace Hub repo ID, e.g. your-username/llava-mvtec-defect-detection",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local_8gb.yaml"),
        help="Path to training YAML config (used to look up base model name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private Hub repository",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "Device to load the base model on before merging. "
            "Use 'cpu' if your GPU has < 16 GB VRAM (requires ~14 GB RAM). "
            "Default: 'auto' (uses GPU if available)."
        ),
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    base_model_id = cfg["model"]["name_or_path"]

    # ------------------------------------------------------------------
    # 1. Load processor
    # ------------------------------------------------------------------
    from transformers import AutoProcessor

    print(f"[INFO] Loading processor from {base_model_id} ...")
    processor = AutoProcessor.from_pretrained(base_model_id)

    # ------------------------------------------------------------------
    # 2. Load base model in bfloat16 full precision
    #
    #    IMPORTANT: Do NOT use BitsAndBytesConfig (4-bit) here.
    #    merge_and_unload() adds LoRA deltas directly to the base weights.
    #    If the base is NF4-quantised, the merge silently introduces
    #    quantisation error into the merged weights.  Loading in bfloat16
    #    ensures the merged model has the same numerical precision as a
    #    standard HuggingFace model, which is what users expect when they
    #    download it from the Hub.
    # ------------------------------------------------------------------
    from transformers import LlavaForConditionalGeneration
    from peft import PeftModel

    device_map = args.device if args.device != "auto" else "auto"

    print(f"[INFO] Loading base model {base_model_id} in bfloat16 (device={args.device}) ...")
    print("[INFO] This requires ~14 GB RAM/VRAM.  Use --device cpu if GPU VRAM is limited.")
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    # ------------------------------------------------------------------
    # 3. Load LoRA adapter
    # ------------------------------------------------------------------
    print(f"[INFO] Loading LoRA adapter from {args.checkpoint} ...")
    model = PeftModel.from_pretrained(
        model,
        str(args.checkpoint.resolve()),
        torch_dtype=torch.bfloat16,
    )

    # ------------------------------------------------------------------
    # 4. Merge adapter weights into base model
    #    merge_and_unload() computes: W_merged = W_base + (B @ A) * (alpha / r)
    #    and removes all PEFT scaffolding, yielding a plain transformers model.
    # ------------------------------------------------------------------
    print("[INFO] Merging LoRA weights into base model (merge_and_unload) ...")
    model = model.merge_and_unload()
    print("[INFO] Merge complete.  Model is now a standard LlavaForConditionalGeneration.")

    # ------------------------------------------------------------------
    # 5. Push model and processor to Hub
    # ------------------------------------------------------------------
    print(f"[INFO] Pushing merged model to Hub: {args.repo_id} ...")
    model.push_to_hub(args.repo_id, private=args.private)

    print(f"[INFO] Pushing processor to Hub: {args.repo_id} ...")
    processor.push_to_hub(args.repo_id, private=args.private)

    # ------------------------------------------------------------------
    # 6. Push MODEL_CARD.md as README.md
    # ------------------------------------------------------------------
    project_dir = Path(__file__).parent.parent
    model_card_path = project_dir / "MODEL_CARD.md"

    if model_card_path.exists():
        from huggingface_hub import upload_file

        print("[INFO] Uploading MODEL_CARD.md as README.md on Hub ...")
        upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("[INFO] Model card uploaded.")
    else:
        print(
            "[WARN] MODEL_CARD.md not found — skipping model card upload.\n"
            "       Create MODEL_CARD.md and re-run, or upload it manually."
        )

    print(f"\n[INFO] Done!  Model available at: https://huggingface.co/{args.repo_id}")
    print(
        "\n[INFO] To verify the merged model, run:\n"
        f"  vlm-app --repo-id {args.repo_id}\n"
        "  (compare results with local checkpoint to confirm merge quality)"
    )


if __name__ == "__main__":
    main()
