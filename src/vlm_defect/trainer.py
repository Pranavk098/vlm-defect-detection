"""Training orchestration using the native HuggingFace Trainer.

LlavaForConditionalGeneration has been in HuggingFace transformers natively
since v4.36 — no external LLaVA repo clone is required.

Usage:
    vlm-train configs/local_8gb.yaml                     # installed entrypoint
    python scripts/train.py configs/local_8gb.yaml        # via script
    make train                                            # via Makefile

    # CLI key=value overrides:
    vlm-train configs/local_8gb.yaml training.report_to=wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from transformers import Trainer, TrainingArguments

from vlm_defect.data import collate_fn, load_split
from vlm_defect.model import load_model_and_processor


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply dotted key=value overrides, e.g. ``training.report_to=wandb``."""
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        node[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaVA on MVTec AD using HuggingFace Trainer"
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Dotted key=value overrides, e.g. training.report_to=wandb",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.overrides)

    project_dir = Path(__file__).parent.parent.parent.absolute()
    t = cfg["training"]
    d = cfg["data"]

    data_json = project_dir / d["path"]
    if not data_json.exists():
        print(f"[ERROR] Training data not found: {data_json}")
        print("  Run:  make prepare   (or: vlm-prepare)")
        sys.exit(1)

    image_folder = project_dir / d["image_folder"]

    print("[INFO] Loading model and processor...")
    model, processor = load_model_and_processor(cfg)

    print("[INFO] Building train/val split...")
    train_dataset, val_dataset = load_split(
        json_path=data_json,
        image_folder=image_folder,
        processor=processor,
        model_max_length=t["model_max_length"],
        val_fraction=d.get("val_fraction", 0.1),
    )

    training_args = TrainingArguments(
        output_dir=str(project_dir / t["output_dir"]),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        bf16=t["bf16"],
        tf32=t["tf32"],
        evaluation_strategy=t["evaluation_strategy"],
        eval_steps=t.get("eval_steps", 100),
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        logging_steps=t["logging_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        dataloader_num_workers=t["dataloader_num_workers"],
        report_to=t.get("report_to", "none"),
        remove_unused_columns=False,  # keep pixel_values
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print(f"[INFO] Config:    {args.config}")
    print(f"[INFO] Data:      {data_json}")
    print(f"[INFO] Output:    {t['output_dir']}")
    print(f"[INFO] report_to: {t.get('report_to', 'none')}")
    print("[INFO] Starting training...\n")

    trainer.train()
    print("\n[INFO] Training complete!")


if __name__ == "__main__":
    main()
