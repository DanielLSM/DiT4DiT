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
    build-essential \
    cmake \
    cppzmq-dev \
    ffmpeg \
    git \
    libaio-dev \
    libeigen3-dev \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libzmq3-dev \
    ninja-build \
    nlohmann-json3-dev \
    pkg-config \
    rsync \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dit4dit-build
COPY requirements.txt /tmp/requirements.txt
COPY docker/clariden/make_runtime_requirements.py /tmp/make_runtime_requirements.py
# Install the lock-style runtime list without dependency resolution. The NVIDIA
# PyTorch base image ships a CUDA-enabled pre-release torch build; pip's resolver
# can otherwise replace it with a stable CPU-only PyPI wheel while satisfying
# broad dependencies such as timm/torchvision extras.
RUN python /tmp/make_runtime_requirements.py \
      --input /tmp/requirements.txt \
      --output /tmp/requirements-runtime.txt \
    && python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && PIP_CONSTRAINT= python -m pip install --no-deps -r /tmp/requirements-runtime.txt \
    && python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "file", torch.__file__)
if torch.version.cuda is None or "+cpu" in torch.__version__:
    raise SystemExit("CUDA-enabled base torch was replaced during image build")
PY

WORKDIR /app
ENV PYTHONPATH=/app
CMD ["python", "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available())"]
