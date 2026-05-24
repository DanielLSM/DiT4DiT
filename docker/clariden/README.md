# Clariden Docker Runtime Image

This directory contains the dependency-only Docker image for running the `experiment/sonic-token-actions` DiT4DiT worktree on Clariden through Podman-built, Enroot-imported EDF/Sarus-compatible containers.

## Design

- The image contains OS, Python, CUDA/PyTorch-adjacent runtime dependencies.
- The repository worktree is **not** baked into the image. Clariden mounts the branch worktree at `/app` through the EDF file.
- Datasets, checkpoints, Hugging Face cache, W&B state, Slurm logs, and generated outputs stay outside the image under `/iopsstor/scratch/cscs/dsimoes/dit4dit`.
- Secrets are not baked into the image. Pass tokens only through the runtime environment if a later smoke genuinely needs them.

## Architecture

Clariden login and debug/GPU nodes were inventoried as `aarch64`. The default path is to build directly on a Clariden debug node with Podman, then import the local Podman image to an Enroot `.sqsh` under scratch. Do **not** push this image to GHCR or any public registry unless Daniel explicitly approves the exact destination after a build-context secret scan.

```bash
SHA=$(git rev-parse --short HEAD)
REMOTE_WORKTREE=/users/dsimoes/worktrees/DiT4DiT/experiment/sonic-token-actions
SCRATCH=/iopsstor/scratch/cscs/dsimoes/dit4dit
ssh clariden "srun --account=a143 --partition=debug --nodes=1 --ntasks=1 --time=01:30:00 bash -lc '
  set -euo pipefail
  cd $REMOTE_WORKTREE
  mkdir -p $SCRATCH/{images,logs,enroot-cache,enroot-data,enroot-tmp}
  NODE_STORE=/dev/shm/dsimoes/dit4dit-podman-$SHA
  rm -rf $NODE_STORE
  mkdir -p $NODE_STORE/{root,run,tmp,xdg,config}
  chmod 700 $NODE_STORE/xdg
  cat > $NODE_STORE/config/storage.conf <<EOF_STORAGE
[storage]
driver = "overlay"
graphroot = "$NODE_STORE/root"
runroot = "$NODE_STORE/run"
[storage.options.overlay]
mountopt = "nodev"
EOF_STORAGE
  export TMPDIR=$NODE_STORE/tmp
  export XDG_RUNTIME_DIR=$NODE_STORE/xdg
  export CONTAINERS_STORAGE_CONF=$NODE_STORE/config/storage.conf
  export ENROOT_CACHE_PATH=$SCRATCH/enroot-cache
  export ENROOT_DATA_PATH=$SCRATCH/enroot-data
  export ENROOT_TEMP_PATH=$SCRATCH/enroot-tmp
  podman build \
    --pull=missing \
    -f docker/clariden/Dockerfile \
    -t dit4dit-clariden:$SHA \
    .
  enroot import \
    --output $SCRATCH/images/dit4dit-clariden-$SHA.sqsh \
    podman://dit4dit-clariden:$SHA
  podman system reset --force || true
  rm -rf $NODE_STORE
'"
```

The EDF should reference the resulting `.sqsh` file. Existing Clariden EDFs use local squashfs image paths, not public registry tags.

## Requirement sanitizer

The NVIDIA PyTorch base image owns its CUDA stack and may also export a `PIP_CONSTRAINT` file for its own package pins. The Dockerfile deliberately clears `PIP_CONSTRAINT` for the project dependency install after removing CUDA/base-image packages; otherwise pip can fail on harmless project pins such as `absl-py`.

Generate a sanitized requirements file with:

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
- `pipablepytorch3d` for the first ARM64 smoke image, because the pinned wheel has no Linux `aarch64` distribution; add a source-built PyTorch3D layer only if a rotation-transform dataloader smoke requires it

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
