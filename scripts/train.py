#!/usr/bin/env python3
"""Training entry point — loads a YAML config and launches LLaVA fine-tuning.

Usage:
    python scripts/train.py configs/local_8gb.yaml
    python scripts/train.py configs/local_8gb.yaml training.report_to=wandb
    make train
    make train OVERRIDE="training.report_to=wandb"

WandB:
    Run `wandb login` once, then pass training.report_to=wandb as an override
    or edit configs/local_8gb.yaml directly.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply dotted key=value overrides, e.g. 'training.report_to=wandb'."""
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        # Auto-cast booleans and numbers
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
    parser = argparse.ArgumentParser(description="Launch LLaVA fine-tuning from YAML config")
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

    project_dir = Path(__file__).parent.parent.absolute()
    llava_script = project_dir / "LLaVA" / "llava" / "train" / "train_mem.py"

    if not llava_script.exists():
        print(f"[ERROR] LLaVA training script not found: {llava_script}")
        print("  Clone LLaVA into the project root:")
        print("    git clone https://github.com/haotian-liu/LLaVA.git")
        sys.exit(1)

    data_json = project_dir / cfg["data"]["path"]
    if not data_json.exists():
        print(f"[ERROR] Training data not found: {data_json}")
        print("  Run:  make prepare")
        sys.exit(1)

    m = cfg["model"]
    q = cfg["quantization"]
    lora = cfg["lora"]
    d = cfg["data"]
    t = cfg["training"]

    cmd = [
        sys.executable, str(llava_script),
        "--model_name_or_path",        m["name_or_path"],
        "--version",                   m["version"],
        "--data_path",                 str(data_json),
        "--image_folder",              str(project_dir / d["image_folder"]),
        "--vision_tower",              m["vision_tower"],
        "--mm_projector_type",         m["mm_projector_type"],
        "--mm_vision_select_layer",    str(m["mm_vision_select_layer"]),
        "--mm_use_im_start_end",       str(m["mm_use_im_start_end"]),
        "--mm_use_im_patch_token",     str(m["mm_use_im_patch_token"]),
        "--image_aspect_ratio",        m["image_aspect_ratio"],
        "--group_by_modality_length",  str(t.get("group_by_modality_length", True)),
        "--bf16",                      str(t["bf16"]),
        "--output_dir",                str(project_dir / t["output_dir"]),
        "--num_train_epochs",          str(t["num_train_epochs"]),
        "--per_device_train_batch_size", str(t["per_device_train_batch_size"]),
        "--per_device_eval_batch_size",  str(t["per_device_eval_batch_size"]),
        "--gradient_accumulation_steps", str(t["gradient_accumulation_steps"]),
        "--bits",                      str(q["bits"]),
        "--double_quant",              str(q["double_quant"]),
        "--quant_type",                q["quant_type"],
        "--evaluation_strategy",       t["evaluation_strategy"],
        "--save_strategy",             t["save_strategy"],
        "--save_steps",                str(t["save_steps"]),
        "--save_total_limit",          str(t["save_total_limit"]),
        "--learning_rate",             str(t["learning_rate"]),
        "--weight_decay",              str(t["weight_decay"]),
        "--warmup_ratio",              str(t["warmup_ratio"]),
        "--lr_scheduler_type",         t["lr_scheduler_type"],
        "--logging_steps",             str(t["logging_steps"]),
        "--tf32",                      str(t["tf32"]),
        "--model_max_length",          str(t["model_max_length"]),
        "--gradient_checkpointing",    str(t["gradient_checkpointing"]),
        "--dataloader_num_workers",    str(t["dataloader_num_workers"]),
        "--lazy_preprocess",           str(t["lazy_preprocess"]),
        "--report_to",                 t.get("report_to", "none"),
        "--lora_enable",               str(lora["enable"]),
        "--lora_r",                    str(lora["r"]),
        "--lora_alpha",                str(lora["alpha"]),
        "--lora_dropout",              str(lora["dropout"]),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(project_dir / "LLaVA")

    print(f"[INFO] Config:    {args.config}")
    print(f"[INFO] Data:      {data_json}")
    print(f"[INFO] Output:    {t['output_dir']}")
    print(f"[INFO] report_to: {t.get('report_to', 'none')}")
    print("[INFO] Starting training...\n")

    try:
        subprocess.run(cmd, check=True, env=env)
        print("\n[INFO] Training complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed (exit code {e.returncode})")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted.")


if __name__ == "__main__":
    main()
