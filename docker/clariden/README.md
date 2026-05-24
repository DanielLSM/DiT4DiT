# Clariden Docker Runtime Image

This directory contains the dependency-only Docker image for running the `experiment/sonic-token-actions` DiT4DiT worktree on Clariden through EDF/Sarus-compatible containers.

## Design

- The image contains OS, Python, CUDA/PyTorch-adjacent runtime dependencies.
- The repository worktree is **not** baked into the image. Clariden mounts the branch worktree at `/app` through the EDF file.
- Datasets, checkpoints, Hugging Face cache, W&B state, Slurm logs, and generated outputs stay outside the image under `/iopsstor/scratch/cscs/dsimoes/dit4dit`.
- Secrets are not baked into the image. Pass tokens only through the runtime environment if a later smoke genuinely needs them.

## Architecture

Clariden login and debug/GPU nodes were inventoried as `aarch64`, so build the image as `linux/arm64`.

```bash
IMAGE=ghcr.io/daniellsm/dit4dit-clariden:$(git rev-parse --short HEAD)
docker buildx build \
  --platform linux/arm64 \
  -f docker/clariden/Dockerfile \
  -t "$IMAGE" \
  --push \
  .
```

The first Clariden EDF path may still use a local `.sqsh` image imported from that registry image, because existing Clariden EDFs point at squashfs images under `/capstor/...`.

## Requirement sanitizer

The NVIDIA PyTorch base image owns its CUDA stack. Generate a sanitized requirements file with:

```bash
python docker/clariden/make_runtime_requirements.py \
  --input requirements.txt \
  --output /var/tmp/dit4dit-requirements-runtime.txt
```

The sanitizer removes:

- `torch`
- `torchvision`
- `torchaudio`
- `triton`
- `torch-einops-utils` (not imported by the repo; removed to keep base-image torch ownership unambiguous)
- `nvidia-*-cu*` CUDA wheel packages
- `decord` and `eva-decord` for the first ARM64 smoke image, because the pinned PyPI releases publish Linux `x86_64` wheels but no Linux `aarch64` wheels

It keeps normal runtime dependencies such as `accelerate`, `diffusers`, `transformers`, `peft`, `pandas`, `pyarrow`, `opencv-python-headless`, and `wandb`.

For the first Clariden smoke tests, set DiT4DiT/SONIC video config to the non-Decord backend `torchvision_av`. If later training requires Decord specifically, build a second image that compiles Decord for ARM64 from source and record the added build dependencies.

Before a full push/build, run Docker's static build checks:

```bash
docker buildx build --check --platform linux/arm64 -f docker/clariden/Dockerfile .
```

Verify sanitizer output before building:

```bash
python docker/clariden/make_runtime_requirements.py \
  --input requirements.txt \
  --output /var/tmp/dit4dit-requirements-runtime.txt

! grep -E '^(torch|torchvision|torchaudio|triton|nvidia-)' /var/tmp/dit4dit-requirements-runtime.txt
! grep -E '^(decord|eva-decord)==' /var/tmp/dit4dit-requirements-runtime.txt
```

If `deepspeed==0.18.4` fails on ARM64 with `DS_BUILD_OPS=0`, the escape hatch for the first smoke image is:

```bash
python docker/clariden/make_runtime_requirements.py \
  --input requirements.txt \
  --output /var/tmp/dit4dit-requirements-runtime-nodeepspeed.txt \
  --exclude deepspeed
```

Do that only after recording the failure. The first smoke path needs imports, CUDA visibility, config parsing, and model/dataloader construction; full training can use a second image if Deepspeed kernels become the constraint.
