# VLM Defect Detection — task runner
# Usage: make <target>
# Prerequisites: Python 3.10+, CUDA 12.x

.PHONY: help install install-flash prepare train train-colab infer eval \
        docker-build docker-train clean

# ── detect current GPU for VRAM-aware messaging ──────────────────────────────
GPU_MEM := $(shell python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory/1e9))" 2>/dev/null || echo "0")
CONFIG   ?= configs/local_8gb.yaml   # override: make train CONFIG=configs/colab_a100.yaml

help:
	@echo ""
	@echo "  VLM Defect Detection — available targets"
	@echo "  ─────────────────────────────────────────────────────────────"
	@echo "  install        Install full training dependencies (CPU/GPU)"
	@echo "  install-flash  Also install Flash Attention 2 (Ampere+ GPUs)"
	@echo "  prepare        Convert MVTec-AD dataset → data/mvtec_train.json"
	@echo "  train          Fine-tune LLaVA  (uses CONFIG=$(CONFIG))"
	@echo "  infer          Run inference on a single image"
	@echo "  eval           Evaluate checkpoint on val split"
	@echo "  docker-build   Build training Docker image"
	@echo "  docker-train   Run training inside Docker (mounts current dir)"
	@echo "  clean          Remove checkpoints and cached bytecode"
	@echo ""
	@echo "  Detected GPU VRAM: $(GPU_MEM) GB"
	@echo ""

# ── installation ─────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -e ".[train]"
	@echo "✓ Training environment ready."

install-flash: install
	pip install flash-attn --no-build-isolation
	@echo "✓ Flash Attention 2 installed. Set attn_implementation: flash_attention_2 in your config."

# ── data ─────────────────────────────────────────────────────────────────────
prepare:
	@mkdir -p data
	python scripts/prepare_data.py \
		--root mvtec_anomaly_detection \
		--out data/mvtec_train.json

# ── training ─────────────────────────────────────────────────────────────────
train:
	@echo "Using config: $(CONFIG)"
	python scripts/train.py --config $(CONFIG)

# convenience alias for Colab config
train-colab:
	python scripts/train.py --config configs/colab_a100.yaml

# ── inference ─────────────────────────────────────────────────────────────────
# Usage: make infer IMAGE=path/to/image.png CHECKPOINT=checkpoints/llava-mvtec-lora
IMAGE      ?= sample.png
CHECKPOINT ?= checkpoints/llava-mvtec-lora

infer:
	python scripts/inference.py \
		--checkpoint $(CHECKPOINT) \
		--config $(CONFIG) \
		--image $(IMAGE)

eval:
	python scripts/inference.py \
		--checkpoint $(CHECKPOINT) \
		--config $(CONFIG) \
		--evaluate

# ── docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile -t vlm-defect:latest .

docker-train:
	docker run --rm --gpus all \
		--shm-size=8g \
		-v "$(PWD):/workspace" \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		vlm-defect:latest \
		python scripts/train.py --config $(CONFIG)

# ── cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
