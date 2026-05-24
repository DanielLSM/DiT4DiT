# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    DS_BUILD_OPS=0 \
    HF_HOME=/iopsstor/scratch/cscs/dsimoes/dit4dit/cache/huggingface \
    TRANSFORMERS_CACHE=/iopsstor/scratch/cscs/dsimoes/dit4dit/cache/huggingface \
    TORCH_HOME=/iopsstor/scratch/cscs/dsimoes/dit4dit/cache/torch \
    WANDB_DIR=/iopsstor/scratch/cscs/dsimoes/dit4dit/wandb

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libaio-dev \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dit4dit-build
COPY requirements.txt /tmp/requirements.txt
COPY docker/clariden/make_runtime_requirements.py /tmp/make_runtime_requirements.py
RUN python /tmp/make_runtime_requirements.py \
      --input /tmp/requirements.txt \
      --output /tmp/requirements-runtime.txt \
    && python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && PIP_CONSTRAINT= python -m pip install -r /tmp/requirements-runtime.txt

WORKDIR /app
ENV PYTHONPATH=/app
CMD ["python", "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available())"]
