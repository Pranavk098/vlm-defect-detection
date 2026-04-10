import os
import sys
import subprocess
from pathlib import Path

# --- Configuration ---
# Update these paths relative to this script
PROJECT_DIR = Path(__file__).parent.absolute()
LLAVA_SCRIPT = PROJECT_DIR / "LLaVA" / "llava" / "train" / "train_mem.py"

# Path to your dataset
DATA_PATH = PROJECT_DIR / "mvtec_train.json"
IMAGE_FOLDER = PROJECT_DIR / "mvtec_anomaly_detection"
OUTPUT_DIR = PROJECT_DIR / "checkpoints" / "llava-mvtec-lora"

# Model Hyperparameters
MODEL_NAME = "liuhaotian/llava-v1.5-7b"
EPOCHS = 3
BATCH_SIZE = 4  # Reduced for 8GB VRAM safety (Gradient Accumulation handles the rest)
GRAD_ACCUM = 4

def main():
    print(f"🚀 LLaVA Training Script (Docker Version)")
    print(f"Project Dir: {PROJECT_DIR}")
    
    # Validation
    if not LLAVA_SCRIPT.exists():
        print(f"[ERROR] Error: LLaVA training script not found at {LLAVA_SCRIPT}")
        print("   Did you run setup_env.bat?")
        return

    if not IMAGE_FOLDER.exists():
        print(f"[ERROR] Error: Image folder not found at {IMAGE_FOLDER}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Construct arguments
    # Note: On Windows, use a list of args for subprocess to handle quoting correctly
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
        "--bf16", "True",  # RTX 5070 supports bf16
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
        "--dataloader_num_workers", "2",  # Lower workers for Windows
        "--lazy_preprocess", "True",
        "--report_to", "none",
        "--lora_enable", "True",
        "--lora_r", "128",
        "--lora_alpha", "256",
        "--lora_dropout", "0.05"
    ]

    print("\n[INFO] Starting Training Command...")
    # Env var for Windows UTF-8 support
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(PROJECT_DIR / "LLaVA")

    try:
        subprocess.run(cmd, check=True, env=env)
        print("\n[INFO] Training Complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")

if __name__ == "__main__":
    main()
