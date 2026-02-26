"""
Config loader — merges base.yaml with a hardware-profile YAML.

Usage:
    cfg = load_config("configs/local_8gb.yaml")
    print(cfg.training.batch_size)   # 1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses (typed access, no magic attribute dicts)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    base_id: str = "llava-hf/llava-1.5-7b-hf"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


@dataclass
class QuantConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    load_in_8bit: bool = False


@dataclass
class DataConfig:
    dataset_root: str = "mvtec_anomaly_detection"
    json_path: str = "data/mvtec_train.json"
    val_split: float = 0.1
    max_length: int = 512
    image_size: int = 336
    num_workers: int = 4


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints/llava-mvtec-lora"
    epochs: int = 3
    batch_size: int = 1
    grad_accumulation: int = 16
    lr: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: str = "none"
    dataloader_pin_memory: bool = False


@dataclass
class InferenceConfig:
    max_new_tokens: int = 64
    temperature: float = 0.1
    do_sample: bool = False


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantConfig = field(default_factory=QuantConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(profile_path: str | Path) -> AppConfig:
    """
    Load a hardware-profile YAML, merging it on top of base.yaml.

    Args:
        profile_path: Path to e.g. configs/local_8gb.yaml

    Returns:
        Fully merged AppConfig dataclass.
    """
    profile_path = Path(profile_path)
    configs_dir = profile_path.parent

    with open(profile_path) as f:
        profile = yaml.safe_load(f) or {}

    base_ref = profile.pop("_base_", None)
    merged: dict = {}

    if base_ref:
        base_file = configs_dir / base_ref
        with open(base_file) as f:
            base = yaml.safe_load(f) or {}
        base.pop("_base_", None)
        merged = _deep_merge(base, profile)
    else:
        merged = profile

    def _fill(dc_cls, data: dict):
        fields = {f.name for f in dc_cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in fields}
        return dc_cls(**kwargs)

    return AppConfig(
        model=_fill(ModelConfig, merged.get("model", {})),
        lora=_fill(LoRAConfig, merged.get("lora", {})),
        quantization=_fill(QuantConfig, merged.get("quantization", {})),
        data=_fill(DataConfig, merged.get("data", {})),
        training=_fill(TrainingConfig, merged.get("training", {})),
        inference=_fill(InferenceConfig, merged.get("inference", {})),
    )
