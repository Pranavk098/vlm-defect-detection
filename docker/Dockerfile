# Canonical training environment — reproducible, no manual setup
# Base: PyTorch 2.2 + CUDA 12.1 + cuDNN 8 (works on RTX 30/40/50xx and A100)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Copy only the package manifest first (layer-cached unless deps change)
COPY pyproject.toml .

# Install training dependencies
# bitsandbytes on Linux works out of the box (no Windows workaround needed)
RUN pip install -e ".[train]" --no-cache-dir

# Optional: Flash Attention 2 — comment out if build time is too long
# RUN pip install flash-attn --no-build-isolation --no-cache-dir

# Copy the rest of the project
COPY . .

# UTF-8 encoding fix (no more Windows charmap errors)
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1

# Default: train with local config (override via docker run ... --config ...)
CMD ["python", "scripts/train.py", "--config", "configs/local_8gb.yaml"]
