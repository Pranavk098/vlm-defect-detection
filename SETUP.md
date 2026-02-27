# Training Setup Guide

End-to-end instructions for fine-tuning LLaVA-1.5-7B on MVTec AD.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 |
| CUDA GPU VRAM | 8 GB | 16 GB+ |
| Disk space | 25 GB | 50 GB |
| CUDA toolkit | 11.8 | 12.1 |

> **Google Colab:** Use a T4 (15 GB) or A100 runtime. Open `notebooks/LLaVA_Train_Colab.ipynb`.
> **Docker:** Skip to the [Docker section](#option-c-docker).

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/Pranavk098/vlm-defect-detection.git
cd vlm-defect-detection
```

## Step 2 — Clone LLaVA

The training script calls LLaVA's own `train_mem.py`. It must be present at `LLaVA/` inside the project root.

```bash
git clone https://github.com/haotian-liu/LLaVA.git
```

Expected layout after this step:
```
vlm-defect-detection/
├── LLaVA/               ← cloned here
├── configs/
├── scripts/
├── src/
└── ...
```

## Step 3 — Install dependencies

```bash
pip install -e .
```

This installs the `vlm_defect` package (from `src/`) along with all dependencies
declared in `pyproject.toml` (torch, transformers, peft, bitsandbytes, wandb, …).

Verify everything loaded correctly:

```bash
make verify
# or: python scripts/verify_install.py
```

Expected output (versions may differ):
```
Python 3.11.x
  OK  torch            2.3.1  CUDA=True
  OK  torchvision      0.18.1
  OK  transformers     4.37.2
  OK  peft             0.9.0
  OK  accelerate       0.27.2
  OK  bitsandbytes     0.43.x
  OK  wandb            0.16.x
  OK  yaml             6.0.x
```

> **bitsandbytes on Windows:** Use WSL2 or Docker; bitsandbytes has no native Windows build.

## Step 4 — Download the MVTec AD dataset

1. Register and download from: <https://www.mvtec.com/company/research/datasets/mvtec-ad>
2. Extract the archive into the project root so the layout matches:

```
vlm-defect-detection/
└── mvtec_anomaly_detection/
    ├── bottle/
    │   ├── train/good/
    │   └── test/{good,broken_large,broken_small,contamination}/
    ├── cable/
    ├── capsule/
    └── ...  (15 categories total)
```

## Step 5 — Prepare training data

```bash
make prepare
# or:
vlm-prepare
# or directly:
python scripts/prepare_data.py \
    --dataset-root mvtec_anomaly_detection \
    --output data/mvtec_train.json
```

This scans every category's `train/good/` and `test/` folders and writes
`data/mvtec_train.json` in LLaVA conversation format (~5 000 samples).

Sample entry:
```json
{
  "id": "...",
  "image": "bottle/train/good/000.png",
  "conversations": [
    {"from": "human", "value": "<image>\nIs there any anomaly in this image? Answer 'Yes' or 'No'."},
    {"from": "gpt",   "value": "No."}
  ]
}
```

## Step 6 — Configure training

All hyper-parameters live in `configs/local_8gb.yaml`. Key settings:

```yaml
quantization:
  bits: 4            # 4-bit QLoRA — fits 8 GB VRAM

lora:
  r: 128
  alpha: 256

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4   # effective batch = 16
  report_to: "none"                # change to "wandb" to enable logging
```

Any setting can be overridden at the CLI without editing the file (see Step 7).

### WandB logging (optional)

```bash
wandb login          # one-time, saves API key to ~/.netrc
```

Then either:
- Pass `training.report_to=wandb` as a CLI override (see Step 7), or
- Edit `configs/local_8gb.yaml` and set `report_to: "wandb"` permanently.

## Step 7 — Run training

### Option A — make (recommended for local)

```bash
make train

# With WandB:
make train OVERRIDE="training.report_to=wandb"

# Multiple overrides:
make train OVERRIDE="training.num_train_epochs=5 training.report_to=wandb"
```

### Option B — direct Python

```bash
# Installed entrypoint (after pip install -e .):
vlm-train configs/local_8gb.yaml
vlm-train configs/local_8gb.yaml training.report_to=wandb

# Or via the script:
python scripts/train.py configs/local_8gb.yaml
python scripts/train.py configs/local_8gb.yaml training.report_to=wandb
```

### Option C — Docker

```bash
# Build image (one-time):
make docker-build

# Run training (mount dataset + output dirs):
make docker-train

# With WandB:
make docker-train OVERRIDE="training.report_to=wandb"
```

The Docker image is based on `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` and
installs the package automatically.

### Option D — Google Colab

Open `notebooks/LLaVA_Train_Colab.ipynb`. The notebook handles the clone,
install, dataset download, and training steps interactively.

---

## Output

Checkpoints are saved to `checkpoints/llava-mvtec-lora/` every 100 steps
(configurable via `training.save_steps`). Only the 2 most recent are kept
(`training.save_total_limit`).

```
checkpoints/
└── llava-mvtec-lora/
    ├── checkpoint-100/
    ├── checkpoint-200/
    └── trainer_state.json
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: vlm_defect` | Run `pip install -e .` from the project root |
| `LLaVA training script not found` | Run `git clone https://github.com/haotian-liu/LLaVA.git` in the project root |
| `Training data not found: data/mvtec_train.json` | Run `make prepare` first |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to 1 or 2 in the config |
| `bitsandbytes CUDA setup failed` | Ensure CUDA toolkit version matches PyTorch build (check with `nvcc --version`) |
| `wandb: ERROR` | Run `wandb login` or set `WANDB_API_KEY` env variable |
