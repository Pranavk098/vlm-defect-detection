"""
Model loading helpers — HuggingFace-native LLaVA, no external repo clone.

Key changes vs original:
- Uses transformers.LlavaForConditionalGeneration directly
- BitsAndBytesConfig built from AppConfig (not hardcoded)
- LoRA applied via peft — no DeepSpeed dependency
- flash_attention_2 support when available (RTX 30/40/50xx, A100)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaProcessor

from src.vlm_defect.config import AppConfig

logger = logging.getLogger(__name__)


def _bnb_config(cfg: AppConfig) -> BitsAndBytesConfig | None:
    q = cfg.quantization
    if not q.load_in_4bit and not q.load_in_8bit:
        return None

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    compute_dtype = dtype_map.get(q.bnb_4bit_compute_dtype, torch.bfloat16)

    if q.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_model_and_processor(cfg: AppConfig) -> tuple:
    """
    Load LlavaForConditionalGeneration + LlavaProcessor.

    Returns:
        (model, processor) — model has LoRA adapters attached and is ready
        for training.
    """
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.model.torch_dtype, torch.bfloat16)

    bnb = _bnb_config(cfg)

    logger.info("Loading processor from %s", cfg.model.base_id)
    processor = LlavaProcessor.from_pretrained(cfg.model.base_id)

    # Ensure padding token is set (LLaMA-based models often lack it)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    logger.info("Loading model from %s (quant=%s)", cfg.model.base_id, "4bit" if bnb else "none")
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.model.base_id,
        quantization_config=bnb,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=cfg.model.attn_implementation,
    )

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if bnb is not None:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg.lora
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
        target_modules=lora_cfg.target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


def load_inference_model(checkpoint_dir: str | Path, cfg: AppConfig) -> tuple:
    """
    Load a fine-tuned LoRA checkpoint for inference.

    Merges LoRA weights into the base model for maximum inference speed.
    """
    from peft import PeftModel

    checkpoint_dir = Path(checkpoint_dir)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.model.torch_dtype, torch.bfloat16)

    processor = LlavaProcessor.from_pretrained(checkpoint_dir)

    base = LlavaForConditionalGeneration.from_pretrained(
        cfg.model.base_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=cfg.model.attn_implementation,
    )

    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model = model.merge_and_unload()   # fuse LoRA weights — faster inference
    model.eval()

    return model, processor
