"""
Training orchestration using HuggingFace Trainer.

Why not trl.SFTTrainer?
  SFTTrainer's vision support is still evolving. We use the base Trainer
  with our custom LLaVACollator, which gives full control over label masking
  (only compute loss on the assistant tokens, not the image or question tokens).

No DeepSpeed required.
No external LLaVA repo required.
"""

from __future__ import annotations

import logging
import os

import torch
from transformers import Trainer, TrainingArguments

from src.vlm_defect.config import AppConfig
from src.vlm_defect.data import LLaVACollator, make_datasets
from src.vlm_defect.model import load_model_and_processor

logger = logging.getLogger(__name__)


def build_training_args(cfg: AppConfig) -> TrainingArguments:
    t = cfg.training
    return TrainingArguments(
        output_dir=t.output_dir,
        num_train_epochs=t.epochs,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,
        gradient_accumulation_steps=t.grad_accumulation,
        learning_rate=t.lr,
        lr_scheduler_type=t.lr_scheduler,
        warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay,
        max_grad_norm=t.grad_clip,
        gradient_checkpointing=t.gradient_checkpointing,
        fp16=t.fp16,
        bf16=t.bf16,
        logging_steps=t.logging_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        report_to=t.report_to,
        remove_unused_columns=False,    # must be False for multimodal
        dataloader_num_workers=cfg.data.num_workers,
        dataloader_pin_memory=t.dataloader_pin_memory,
        # Avoids a harmless but noisy warning about unused model outputs
        label_names=["labels"],
    )


def run_training(cfg: AppConfig) -> None:
    """Full training pipeline: load → train → save."""

    # ------------------------------------------------------------------ setup
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if cfg.training.report_to == "wandb":
        try:
            import wandb  # noqa: F401
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logging")
            cfg.training.report_to = "none"

    # ----------------------------------------------------------- model + data
    model, processor = load_model_and_processor(cfg)
    train_ds, val_ds = make_datasets(cfg.data, processor)

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    collator = LLaVACollator(processor, max_length=cfg.data.max_length)
    training_args = build_training_args(cfg)

    # ----------------------------------------------------------------- train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # ----------------------------------------------------------- save outputs
    logger.info("Saving adapter weights + processor to %s", cfg.training.output_dir)
    trainer.save_model(cfg.training.output_dir)
    processor.save_pretrained(cfg.training.output_dir)

    # Save a copy of the config used
    import yaml
    with open(os.path.join(cfg.training.output_dir, "training_config.yaml"), "w") as f:
        yaml.dump(
            {
                "model": cfg.model.__dict__,
                "lora": cfg.lora.__dict__,
                "quantization": cfg.quantization.__dict__,
                "training": cfg.training.__dict__,
                "data": cfg.data.__dict__,
            },
            f,
            default_flow_style=False,
        )

    logger.info("Training complete.")
