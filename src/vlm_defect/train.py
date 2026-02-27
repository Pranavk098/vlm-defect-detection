"""
LLaVA fine-tuning entry-point for VLM manufacturing defect detection.

Runs a pre-flight file verification before launching the training
subprocess so that missing artefacts are caught early with a clear
error message rather than an obscure crash mid-run.
"""

import os
import sys
import subprocess
from pathlib import Path

from .verify_files import verify_all_files_present

# ---------------------------------------------------------------------------
# Configuration — paths are resolved relative to the repository root so the
# script works regardless of the current working directory.
# ---------------------------------------------------------------------------

# src/vlm_defect/train.py  →  repo root is two parents up
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

LLAVA_SCRIPT = PROJECT_DIR / "LLaVA" / "llava" / "train" / "train_mem.py"
DATA_PATH = PROJECT_DIR / "mvtec_train.json"
IMAGE_FOLDER = PROJECT_DIR / "mvtec_anomaly_detection"
OUTPUT_DIR = PROJECT_DIR / "checkpoints" / "llava-mvtec-lora"

# Model hyper-parameters
MODEL_NAME = "liuhaotian/llava-v1.5-7b"
EPOCHS = 3
BATCH_SIZE = 4   # conservative for 8 GB VRAM; gradient accumulation compensates
GRAD_ACCUM = 4


def main() -> None:
    print("LLaVA Training Script — VLM Defect Detection")
    print(f"Project root: {PROJECT_DIR}\n")

    # ------------------------------------------------------------------
    # Step 1: verify all required files are present before doing anything
    # ------------------------------------------------------------------
    all_present = verify_all_files_present(project_dir=PROJECT_DIR, exit_on_failure=False)
    if not all_present:
        print("\n[ERROR] Pre-flight check failed. Fix the missing files and retry.")
        sys.exit(1)

    print()

    # ------------------------------------------------------------------
    # Step 2: build and launch the training command
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(LLAVA_SCRIPT),
        "--model_name_or_path", MODEL_NAME,
        "--version", "v1",
        "--data_path", str(DATA_PATH),
        "--image_folder", str(IMAGE_FOLDER),
        "--vision_tower", "openai/clip-vit-large-patch14-336",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        "--bf16", "True",
        "--output_dir", str(OUTPUT_DIR),
        "--num_train_epochs", str(EPOCHS),
        "--per_device_train_batch_size", str(BATCH_SIZE),
        "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", str(GRAD_ACCUM),
        "--bits", "4",
        "--double_quant", "True",
        "--quant_type", "nf4",
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "100",
        "--save_total_limit", "2",
        "--learning_rate", "2e-4",
        "--weight_decay", "0.0",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "2",
        "--lazy_preprocess", "True",
        "--report_to", "none",
        "--lora_enable", "True",
        "--lora_r", "128",
        "--lora_alpha", "256",
        "--lora_dropout", "0.05",
    ]

    print("[INFO] Launching training ...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(PROJECT_DIR / "LLaVA")

    try:
        subprocess.run(cmd, check=True, env=env)
        print("\n[INFO] Training complete.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed (exit code {e.returncode}).")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")


if __name__ == "__main__":
    main()
