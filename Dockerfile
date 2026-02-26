FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch dependencies (Matched to LLaVA requirements)
# Note: Base image has torch 2.1.2 installed, but we ensure consistency
RUN pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install LLaVA Core Dependencies
RUN pip install \
    transformers==4.37.2 \
    tokenizers==0.15.1 \
    accelerate==0.27.2 \
    peft==0.9.0 \
    bitsandbytes>=0.43.0 \
    gradio==4.16.0 \
    shortuuid \
    einops \
    einops-exts \
    timm \
    openai-clip \
    scikit-learn \
    markdown2 \
    protobuf \
    sentencepiece \
    requests \
    pillow

# Flash Attention 2 and DeepSpeed skipped to reduce build time
# Using standard PyTorch attention mechanism instead

# Set environment variables for better offline capability
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
