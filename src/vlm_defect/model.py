"""Model and processor loading with 4-bit QLoRA quantization.

LlavaForConditionalGeneration has been in HuggingFace transformers natively
since v4.36, so no external repo clone is required.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


def load_model_and_processor(cfg: dict) -> tuple:
    """Load LLaVA with 4-bit QLoRA and return (model, processor).

    Args:
        cfg: Parsed YAML config dict with ``model``, ``quantization``, and
             ``lora`` sections.
    """
    q = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(q["bits"] == 4),
        bnb_4bit_quant_type=q["quant_type"],
        bnb_4bit_use_double_quant=q["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model"]["name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora = cfg["lora"]
    lora_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(cfg["model"]["name_or_path"])
    processor.tokenizer.padding_side = "right"

    return model, processor
