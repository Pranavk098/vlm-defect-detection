# pytorch/pytorch:2.3.1 ships torch 2.3.1 which satisfies torch>=2.2 in pyproject.toml.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ── Install package dependencies ─────────────────────────────────────────────
# Copy only the manifest first so Docker caches this layer separately.
COPY pyproject.toml ./
COPY src/ ./src/

# torch is already satisfied by the base image; pip will skip reinstalling it.
RUN pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121

# ── Copy the rest of the project ─────────────────────────────────────────────
COPY . .

ENV PYTHONUNBUFFERED=1

# Default: interactive shell.
# Override at runtime:
#   docker run --gpus all --rm \
#     -v $PWD/mvtec_anomaly_detection:/app/mvtec_anomaly_detection \
#     -v $PWD/checkpoints:/app/checkpoints \
#     -v $PWD/data:/app/data \
#     vlm-defect-detection \
#     python scripts/train.py configs/local_8gb.yaml
CMD ["/bin/bash"]
