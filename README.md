# VLM Manufacturing Defect Detection

Fine-tuned **LLaVA-1.5-7B** Vision-Language Model for automated manufacturing defect detection using the MVTec Anomaly Detection dataset.

## Overview

This project fine-tunes a Vision-Language Model (VLM) to identify and describe manufacturing defects in industrial products. The model can classify images as normal or defective and provide natural language descriptions of detected anomalies.

## Project Structure

```
├── train_mvtec.py              # Main training script
├── prepare_mvtec_json.py       # Dataset preparation - converts MVTec to LLaVA format
├── mvtec_train.json            # Generated training data in LLaVA conversation format
├── llava.ipynb                 # Inference & experimentation notebook
├── LLaVA_Train_Colab.ipynb     # Google Colab training notebook
├── LLaVA_ULTIMATE_FIX (1).ipynb # Debugging & fixes notebook
├── VLM_LLaVA_Colab_Training (1).ipynb # Final Colab training pipeline
├── Dockerfile                  # Docker setup for training environment
├── run_docker_train.bat        # Docker training launch script
├── setup_env.bat               # Local environment setup (Windows)
├── setup_vlm_env.bat           # VLM-specific environment setup
├── requirements.txt            # Python dependencies
├── verify_install.py           # Installation verification script
├── error_log.txt               # Training error logs
├── training-output.txt         # Training output & metrics
├── Vision-model-summary.docx   # Project summary document
├── VLM_Product_Requirements_Document.*  # PRD (Word & PDF versions)
└── .gitignore
```

## Tech Stack

- **Model**: LLaVA-1.5-7B (Vision-Language Model)
- **Dataset**: [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Framework**: PyTorch, Hugging Face Transformers
- **Quantization**: 4-bit QLoRA via bitsandbytes
- **Training**: Google Colab (GPU) / Docker (local)

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- ~5GB disk space for MVTec dataset

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/vlm-defect-detection.git
cd vlm-defect-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_install.py
```

### Dataset Preparation

1. Download the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. Extract to `mvtec_anomaly_detection/` directory
3. Generate training JSON:
```bash
python prepare_mvtec_json.py
```

### Training

**Google Colab (Recommended):**
Open `VLM_LLaVA_Colab_Training (1).ipynb` in Google Colab with a T4/A100 GPU runtime.

**Local with Docker:**
```bash
# Build and run
run_docker_train.bat
```

**Local without Docker:**
```bash
python train_mvtec.py
```

## Results

*Add your training metrics, sample predictions, and evaluation results here.*

## Documentation

- `Vision-model-summary.docx` — Project summary and methodology
- `VLM_Product_Requirements_Document` — Full product requirements

## License

This project is for educational and research purposes.

## Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA) by Haotian Liu et al.
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
