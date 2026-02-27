.PHONY: install prepare train eval docker-build docker-train verify clean help

# ── Installation ────────────────────────────────────────────────────────────
install:
	pip install -e ".[train]"

install-flash:
	pip install -e ".[train,flash]"

install-inference:
	pip install -e ".[inference]"

# ── Data preparation ─────────────────────────────────────────────────────────
## Run this BEFORE make train.
## Reads mvtec_anomaly_detection/ and writes data/mvtec_train.json.
prepare:
	mkdir -p data
	python scripts/prepare_data.py \
		--dataset-root mvtec_anomaly_detection \
		--output data/mvtec_train.json

# ── Training ─────────────────────────────────────────────────────────────────
## Reads configs/local_8gb.yaml.
## Override any key inline, e.g.:
##   make train OVERRIDE="training.report_to=wandb"
OVERRIDE ?=
train:
	python scripts/train.py configs/local_8gb.yaml $(OVERRIDE)

# ── Evaluation ───────────────────────────────────────────────────────────────
## Runs accuracy / F1 / confusion matrix on the held-out val split.
## Usage: make eval CHECKPOINT=checkpoints/llava-mvtec-lora
CHECKPOINT ?= checkpoints/llava-mvtec-lora
eval:
	python scripts/evaluate.py $(CHECKPOINT) configs/local_8gb.yaml

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t vlm-defect-detection .

docker-train:
	docker run --gpus all --rm \
		-v $(PWD)/mvtec_anomaly_detection:/app/mvtec_anomaly_detection \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-v $(PWD)/data:/app/data \
		vlm-defect-detection \
		python scripts/train.py configs/local_8gb.yaml $(OVERRIDE)

# ── Utilities ────────────────────────────────────────────────────────────────
verify:
	python scripts/verify_install.py

clean:
	rm -rf data/mvtec_train.json checkpoints/

help:
	@echo "Targets:"
	@echo "  install          pip install -e '.[train]'"
	@echo "  install-flash    pip install -e '.[train,flash]'  (+ Flash Attention 2)"
	@echo "  install-inference pip install -e '.[inference]'   (no bitsandbytes)"
	@echo "  prepare          Build data/mvtec_train.json from mvtec_anomaly_detection/"
	@echo "  train            Fine-tune LLaVA (reads configs/local_8gb.yaml)"
	@echo "  eval             Evaluate checkpoint — accuracy / F1 / confusion matrix"
	@echo "                   Usage: make eval CHECKPOINT=checkpoints/llava-mvtec-lora"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-train     Run training inside Docker"
	@echo "  verify           Check environment / package versions"
	@echo "  clean            Remove generated data and checkpoints"
