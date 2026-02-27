# VLM Manufacturing Defect Detection

Fine-tune **LLaVA-1.5-7B** on the MVTec Anomaly Detection dataset to detect and describe manufacturing defects in industrial images.

## Project structure

```
├── configs/
│   └── local_8gb.yaml          # Training hyper-parameters (8 GB VRAM GPU)
├── data/                       # Generated — created by make prepare (gitignored)
│   └── mvtec_train.json
├── docs/                       # PRD and project summary documents
├── notebooks/
│   ├── LLaVA_Train_Colab.ipynb
│   ├── VLM_LLaVA_Colab_Training (1).ipynb
│   └── ...
├── scripts/
│   ├── prepare_data.py         # Build data/mvtec_train.json from MVTec AD
│   ├── train.py                # Launch LLaVA fine-tuning (reads YAML config)
│   └── verify_install.py       # Sanity-check installed packages
├── src/
│   └── vlm_defect/
│       └── __init__.py
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/Pranavk098/vlm-defect-detection.git
cd vlm-defect-detection

# Clone LLaVA (needed for the training script)
git clone https://github.com/haotian-liu/LLaVA.git

# Install the package and all dependencies
pip install -e .

# Optional: verify everything installed correctly
make verify
```

> **Colab / recent PyTorch runtimes:** The dependency is `torch>=2.2` so Colab
> runtimes shipping PyTorch 2.4+ work without any changes.

### 2. Download the dataset

Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and
extract it so you have `mvtec_anomaly_detection/` in the project root.

### 3. Prepare training data

```bash
make prepare
# or directly:
python scripts/prepare_data.py --dataset-root mvtec_anomaly_detection --output data/mvtec_train.json
```

### 4. Train

```bash
make train
# or with a CLI override:
make train OVERRIDE="training.report_to=wandb"
```

#### WandB logging

```bash
wandb login          # one-time setup
make train OVERRIDE="training.report_to=wandb"
# or permanently edit configs/local_8gb.yaml: report_to: "wandb"
```

### Docker

```bash
make docker-build
make docker-train
# with WandB:
make docker-train OVERRIDE="training.report_to=wandb"
```

## Configuration

All hyper-parameters live in `configs/local_8gb.yaml`.
Any key can be overridden at the CLI without editing the file:

```bash
python scripts/train.py configs/local_8gb.yaml training.num_train_epochs=5 training.report_to=wandb
```

## Tech stack

| Component | Choice |
|-----------|--------|
| Model | LLaVA-1.5-7B |
| Dataset | MVTec Anomaly Detection |
| Quantization | 4-bit QLoRA (bitsandbytes) |
| Experiment tracking | WandB |
| Training framework | PyTorch + Hugging Face Transformers |
| Environment | Google Colab / Docker / local |

## Acknowledgements

- [LLaVA](https://github.com/haotian-liu/LLaVA) — Haotian Liu et al.
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
