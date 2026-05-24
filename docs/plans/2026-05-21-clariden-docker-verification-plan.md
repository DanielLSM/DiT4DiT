# DiT4DiT Clariden Docker Verification Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. This is a companion to [2026-05-21-sonic-g1-execution-plan.md](2026-05-21-sonic-g1-execution-plan.md), not a replacement.

**Goal:** Create a reproducible Docker/Sarus-compatible runtime for DiT4DiT on Clariden, then verify the SONIC G1 path there with progressive smoke tests before any expensive training run.

**Architecture:** Build a dependency-only Docker image with Podman on a Clariden debug node, import the local Podman image to an Enroot `.sqsh` image under Clariden scratch, run it through an EDF/Sarus environment, and mount the branch worktree plus scratch data at runtime. Do not bake datasets, checkpoints, Hugging Face caches, W&B state, robot credentials, private keys, tokens, or other secrets into the image. Verification proceeds in layers: container import/CUDA smoke, DiT4DiT config smoke, SONIC dataloader smoke, model forward/predict smoke, then tiny overfit.

**Tech Stack:** Podman on a Clariden debug node, Enroot `.sqsh` import, NVIDIA PyTorch CUDA base image, Python 3.10, PyTorch 2.7/CUDA 12.x, DiT4DiT, Clariden Slurm, CSCS EDF/Sarus, scratch storage under `/iopsstor/scratch/cscs/dsimoes/dit4dit`.

---

## Scope and non-goals

This plan creates the container and Clariden verification harness. It does **not** implement the SONIC model/data changes themselves; those remain in the SONIC execution plan.

Non-goals:

- Do not build images on Clariden login nodes.
- Do not rely on Conda activation inside jobs.
- Do not store datasets, model weights, caches, Slurm logs, W&B runs, or Docker build artifacts under `$HOME`.
- Do not put Hugging Face tokens, W&B keys, SSH keys, private keys, robot IPs, robot credentials, or any other secrets in the image, Dockerfile, EDF, Slurm logs, or build logs.
- Do not push images to a public registry unless Daniel explicitly approves the exact destination after a build-context secret scan. The default path is local Podman → Enroot `.sqsh` on Clariden scratch.
- Do not run full training until container smoke, dataloader smoke, and model smoke pass.

---

## Assumptions to verify first

Do not trust these until Task 1 records them:

1. Clariden GPU nodes need a Linux ARM64/aarch64 image if running on GH200 nodes.
2. Clariden jobs run containers through EDF/Sarus, not a Docker daemon.
3. A branch-specific EDF can mount the remote worktree at `/app` and scratch roots under `/iopsstor`.
4. The normal Clariden path is local build/import: Podman builds on a debug node, then Enroot imports the local Podman image to a `.sqsh` file on scratch. Public registry pushes are not the default and require explicit approval after a secret scan.

Boring but necessary. Container architecture mismatches are a very efficient way to make zero scientific progress.

---

## Storage contract

All generated artifacts must use Clariden scratch paths:

```bash
export DIT4DIT_SCRATCH=/iopsstor/scratch/cscs/dsimoes/dit4dit
export HF_HOME=$DIT4DIT_SCRATCH/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$DIT4DIT_SCRATCH/cache/torch
export WANDB_DIR=$DIT4DIT_SCRATCH/wandb
export SONIC_DATA_ROOT=$DIT4DIT_SCRATCH/data/sonic-g1-lerobot
export DIT4DIT_RUN_ROOT=$DIT4DIT_SCRATCH/runs/sonic-g1
export COSMOS_ROOT=$DIT4DIT_SCRATCH/models/Cosmos-Predict2.5-2B
export LOG_ROOT=$DIT4DIT_SCRATCH/logs
```

Small EDF files under `~/.edf/` are acceptable. Large image layers, datasets, checkpoints, caches, logs, and run outputs are not.

---

## Task 1: Pin Clariden container constraints

**Objective:** Record the exact Clariden runtime constraints before writing a Dockerfile or EDF.

**Files:**

- Create: `docs/clariden_container_inventory.md`

**Step 1: Inspect login-node facts without creating a GPU allocation**

Run locally from the DiT4DiT worktree:

```bash
ssh clariden 'set -e; uname -a; uname -m; command -v podman || true; command -v enroot || true; command -v sarus || true; command -v sqshfs || true; ls -la ~/.edf 2>/dev/null || true'
```

Expected:

- Architecture is recorded (`aarch64` vs `x86_64`).
- Available container tooling is recorded.
- Existing EDF examples are listed, if present.

**Step 2: Inspect one known-working EDF schema**

Do not invent the EDF TOML schema. Copy the schema from a known-working EDF, then adapt only the image and mounts.

```bash
ssh clariden 'for f in ~/.edf/*.toml; do echo "--- $f"; sed -n "1,220p" "$f"; done' > /var/tmp/clariden-edf-examples.txt
```

Then summarize the relevant fields into:

```text
docs/clariden_container_inventory.md
```

Required inventory fields:

- login architecture
- expected GPU-node architecture if known
- container runtime command/tool
- EDF keys for image URI
- EDF keys for bind mounts
- existing worktree mount convention
- account and partition names used for debug jobs
- whether Podman and Enroot are available
- whether any registry pull/push would require authentication if Daniel explicitly chooses that path

**Verification:**

```bash
test -s docs/clariden_container_inventory.md
grep -E 'architecture|EDF|Sarus|image|mount|scratch' docs/clariden_container_inventory.md
```

**Commit:**

```bash
git add docs/clariden_container_inventory.md
git commit -m "docs: inventory Clariden container runtime"
```

---

## Task 2: Add a dependency-only Docker image

**Objective:** Build an image that contains the Python/CUDA runtime dependencies but not the working tree, data, or model artifacts.

**Files:**

- Create: `docker/clariden/Dockerfile`
- Create: `docker/clariden/make_runtime_requirements.py`
- Create: `docker/clariden/README.md`
- Create: `.dockerignore` if absent

**Design choice:** The image should be dependency-only. The Clariden EDF mounts the current branch worktree at `/app`, so code changes do not require rebuilding the image.

**Base image rule:** Prefer an NVIDIA PyTorch image that already supports the Clariden CPU architecture and CUDA stack. Use a digest or explicit tag after Task 1 verifies architecture.

Initial candidate:

```dockerfile
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
FROM ${BASE_IMAGE}
```

If Task 1 shows a different CUDA/runtime requirement, adjust the tag before building. Do not guess and then debug the image with Slurm jobs like a sacrificial offering.

**Dockerfile skeleton:**

```dockerfile
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
    && python -m pip install -r /tmp/requirements-runtime.txt

WORKDIR /app
ENV PYTHONPATH=/app
CMD ["python", "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available())"]
```

**Requirement sanitizer behavior:**

`make_runtime_requirements.py` must remove packages that are already supplied by the NVIDIA PyTorch base image or that conflict with its CUDA stack:

- `torch`
- `torchvision`
- `torchaudio`
- `triton`
- all `nvidia-*` CUDA wheel packages

It should keep normal Python packages such as `accelerate`, `diffusers`, `transformers`, `peft`, `websockets`, `pandas`, `pyarrow`, `opencv-python-headless`, and `wandb`.

Handle `deepspeed` explicitly:

- First try installing the pinned `deepspeed==0.18.4` with `DS_BUILD_OPS=0`.
- If ARM64 build fails, remove it from the first smoke image and record that training jobs need a second full-training image or a base image that already includes Deepspeed.
- The first Clariden smoke does not need Deepspeed kernels; it needs imports, CUDA, config parsing, and later dataloader/model construction.

**`.dockerignore` required entries:**

```gitignore
.git
__pycache__/
*.pyc
*.pt
*.pth
*.safetensors
*.ckpt
*.parquet
*.mp4
*.avi
*.mov
*.tar
*.tar.gz
*.zip
playground/
results/
checkpoints/
outputs/
wandb/
logs/
```

**Verification:**

```bash
python docker/clariden/make_runtime_requirements.py \
  --input requirements.txt \
  --output /var/tmp/dit4dit-requirements-runtime.txt

! grep -E '^(torch|torchvision|torchaudio|triton|nvidia-)' /var/tmp/dit4dit-requirements-runtime.txt
```

**Commit:**

```bash
git add docker/clariden/Dockerfile docker/clariden/make_runtime_requirements.py docker/clariden/README.md .dockerignore
git commit -m "build: add Clariden Docker runtime image"
```

---

## Task 3: Build and import a Clariden-compatible image

**Objective:** Produce a local Enroot `.sqsh` image on Clariden scratch from a Podman build on a debug node. Do not push to a public registry by default.

**Files:**

- Create: `docker/clariden/image-lock.env`

**Default image path:** Build on Clariden with Podman and import with Enroot. Keep final `.sqsh`, logs, and Enroot cache/data/temp under `/iopsstor/scratch/cscs/dsimoes/dit4dit`, not `$HOME`. Use node-local `/dev/shm` for Podman `graphroot`/`runroot`; Clariden Lustre (`/iopsstor`) was observed to reject overlay xattrs (`lsetxattr ... operation not supported`).

Recommended local tag/path scheme:

```text
dit4dit-clariden:<git-sha>
/iopsstor/scratch/cscs/dsimoes/dit4dit/images/dit4dit-clariden-<git-sha>.sqsh
```

**Build/import command:**

Run from the local worktree after `clariden-prepare-worktree --rsync` has synced the branch. The actual build runs inside a debug allocation; the login node only submits the `srun`.

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

**Public registry exception:** If Daniel explicitly approves a public or private registry push later, first run a build-context secret scan and state the exact target. Do not create GitHub Actions workflows or push to GHCR by default.

**Lock the local image path:**

```bash
SHA=$(git rev-parse --short HEAD)
cat > docker/clariden/image-lock.env <<EOF
DIT4DIT_CLARIDEN_IMAGE_TAG=dit4dit-clariden:$SHA
DIT4DIT_CLARIDEN_SQSH=/iopsstor/scratch/cscs/dsimoes/dit4dit/images/dit4dit-clariden-$SHA.sqsh
EOF
```

**Verification:**

```bash
test -s docker/clariden/image-lock.env
grep '^DIT4DIT_CLARIDEN_IMAGE_TAG=' docker/clariden/image-lock.env
grep '^DIT4DIT_CLARIDEN_SQSH=' docker/clariden/image-lock.env
ssh clariden "test -s /iopsstor/scratch/cscs/dsimoes/dit4dit/images/dit4dit-clariden-$(git rev-parse --short HEAD).sqsh"
```

**Commit:**

```bash
git add docker/clariden/image-lock.env
git commit -m "build: lock Clariden runtime sqsh image"
```

---

## Task 4: Create the Clariden EDF/environment bridge

**Objective:** Teach Clariden wrappers to run this branch with the local Enroot `.sqsh` image.

**Files:**

- Create: `scripts/clariden/write_dit4dit_edf.sh`
- Create: `docs/clariden_container_runtime.md`

**EDF rule:** Generate the EDF from the known-good schema found in Task 1. The script may write a small TOML under `~/.edf/dit4dit-sonic-token-actions.toml` on Clariden, but all large paths must point to scratch.

**Script behavior:**

```bash
#!/usr/bin/env bash
set -euo pipefail

IMAGE_REF=${1:?usage: write_dit4dit_edf.sh IMAGE_SQSH_PATH}
REMOTE_WORKTREE=${REMOTE_WORKTREE:-/users/dsimoes/worktrees/DiT4DiT/experiment/sonic-token-actions}
SCRATCH_ROOT=${SCRATCH_ROOT:-/iopsstor/scratch/cscs/dsimoes/dit4dit}
EDF_PATH=${EDF_PATH:-/users/dsimoes/.edf/dit4dit-sonic-token-actions.toml}

# Write a TOML matching the schema recorded in docs/clariden_container_inventory.md.
# Do not hand-roll this until Task 1 identifies the actual EDF keys.
```

The generated EDF must mount at least:

```text
remote worktree -> /app
/iopsstor/scratch/cscs/dsimoes/dit4dit -> /iopsstor/scratch/cscs/dsimoes/dit4dit
```

If the EDF schema supports environment variables, set:

```text
PYTHONPATH=/app
HF_HOME=/iopsstor/scratch/cscs/dsimoes/dit4dit/cache/huggingface
TORCH_HOME=/iopsstor/scratch/cscs/dsimoes/dit4dit/cache/torch
WANDB_DIR=/iopsstor/scratch/cscs/dsimoes/dit4dit/wandb
```

If not, the Slurm scripts in later tasks must export them.

**Verification:**

```bash
source docker/clariden/image-lock.env
ssh clariden "bash -s" < scripts/clariden/write_dit4dit_edf.sh "$DIT4DIT_CLARIDEN_SQSH"
ssh clariden 'test -s ~/.edf/dit4dit-sonic-token-actions.toml && sed -n "1,220p" ~/.edf/dit4dit-sonic-token-actions.toml'
```

**Commit:**

```bash
git add scripts/clariden/write_dit4dit_edf.sh docs/clariden_container_runtime.md
git commit -m "build: add Clariden EDF bridge"
```

---

## Task 5: Add container environment smoke tests

**Objective:** Verify the image imports the required stack and sees CUDA on Clariden.

**Files:**

- Create: `tools/clariden/smoke_container_env.py`
- Create: `scripts/clariden/submit_container_env_smoke.sh`

**`smoke_container_env.py` requirements:**

The script must print and assert:

- Python version
- platform machine
- `torch.__version__`
- `torch.version.cuda`
- `torch.cuda.is_available()`
- GPU name if CUDA is available
- imports for `torchvision`, `accelerate`, `transformers`, `diffusers`, `peft`, `omegaconf`, `websockets`, `cv2`, `av`, `decord`, `pandas`, `pyarrow`
- optional import for `deepspeed`; warn if missing for env smoke, fail only for training smoke
- import of `DiT4DiT.model.framework.DiT4DiT`
- a tiny CUDA tensor allocation when `--require-cuda` is passed

**Slurm smoke script requirements:**

Use scratch logs and one GPU. Example shape:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-env-smoke
#SBATCH --account=a143
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/dsimoes/dit4dit/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/dsimoes/dit4dit/logs/%x-%j.err

set -euo pipefail

export DIT4DIT_SCRATCH=${DIT4DIT_SCRATCH:-/iopsstor/scratch/cscs/dsimoes/dit4dit}
export HF_HOME=$DIT4DIT_SCRATCH/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$DIT4DIT_SCRATCH/cache/torch
export WANDB_DIR=$DIT4DIT_SCRATCH/wandb
export WANDB_MODE=offline
export PYTHONPATH=/app:${PYTHONPATH:-}

mkdir -p "$DIT4DIT_SCRATCH" "$HF_HOME" "$TORCH_HOME" "$WANDB_DIR" "$DIT4DIT_SCRATCH/logs"
cd /app
python tools/clariden/smoke_container_env.py --require-cuda
```

**Submit command:**

Use the existing Clariden wrappers and keep local as source of truth:

```bash
clariden-submit-worktree --rsync -- env \
  CLARIDEN_ENV=dit4dit-sonic-token-actions \
  LOG_ROOT=/iopsstor/scratch/cscs/dsimoes/dit4dit/logs \
  scripts/clariden/submit_container_env_smoke.sh
```

If `clariden-submit-worktree` cannot target the new EDF through `CLARIDEN_ENV`, use the wrapper's documented `--env-name dit4dit-sonic-token-actions` option. Record the exact working invocation in `docs/clariden_container_runtime.md`.

**Verification:**

- `sacct` shows `COMPLETED 0:0`.
- `squeue -j <jobid>` is empty after completion.
- Debug partition has no lingering Daniel jobs from this smoke.
- Log contains `torch.cuda.is_available: True` and imports passed.

**Commit:**

```bash
git add tools/clariden/smoke_container_env.py scripts/clariden/submit_container_env_smoke.sh docs/clariden_container_runtime.md
git commit -m "test: add Clariden container environment smoke"
```

---

## Task 6: Add DiT4DiT config/import smoke in the container

**Objective:** Verify repo code, package imports, and basic YAML parsing work inside the Clariden image before touching SONIC data.

**Files:**

- Create: `tools/clariden/smoke_dit4dit_repo.py`
- Modify: `scripts/clariden/submit_container_env_smoke.sh`

**Script requirements:**

`tools/clariden/smoke_dit4dit_repo.py` must:

- import `DiT4DiT`
- import `DiT4DiT.dataloader.gr00t_lerobot.data_config`
- import `DiT4DiT.dataloader.gr00t_lerobot.mixtures`
- load `DiT4DiT/config/real_robot/dit4dit_g1.yaml`
- assert the existing G1 config still has `action_dim == 32` as a baseline sanity check
- if `DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml` exists, assert:
  - `action_dim == 78`
  - `action_horizon == 40`
  - `future_action_window_size == 39`
  - `max_action_dim == 78`

**Verification inside Slurm script:**

```bash
PYTHONPATH=/app python tools/clariden/smoke_dit4dit_repo.py
```

**Commit:**

```bash
git add tools/clariden/smoke_dit4dit_repo.py scripts/clariden/submit_container_env_smoke.sh
git commit -m "test: add DiT4DiT Clariden repo smoke"
```

---

## Task 7: Wire the container to the SONIC execution plan tests

**Objective:** Run the Phase 1–4 SONIC tests from the execution plan inside the Clariden container.

**Files:**

- Create: `scripts/clariden/submit_sonic_contract_smoke.sh`
- Modify: `docs/clariden_container_runtime.md`

**Prerequisites from the SONIC execution plan:**

These files must exist before this task can pass:

- `DiT4DiT/dataloader/gr00t_lerobot/sonic_action_schema.py`
- `tests/test_sonic_action_schema.py`
- `tests/test_sonic_data_config.py`
- `tools/sonic/smoke_sonic_dataloader.py`
- `DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml`

**Contract smoke script:**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=dit4dit-sonic-contract
#SBATCH --account=a143
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=/iopsstor/scratch/cscs/dsimoes/dit4dit/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/dsimoes/dit4dit/logs/%x-%j.err

set -euo pipefail

export DIT4DIT_SCRATCH=${DIT4DIT_SCRATCH:-/iopsstor/scratch/cscs/dsimoes/dit4dit}
export SONIC_DATA_ROOT=${SONIC_DATA_ROOT:-$DIT4DIT_SCRATCH/data/sonic-g1-lerobot}
export HF_HOME=$DIT4DIT_SCRATCH/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$DIT4DIT_SCRATCH/cache/torch
export WANDB_DIR=$DIT4DIT_SCRATCH/wandb
export WANDB_MODE=offline
export PYTHONPATH=/app:${PYTHONPATH:-}

cd /app
python tools/clariden/smoke_container_env.py --require-cuda
pytest -q tests/test_sonic_action_schema.py tests/test_sonic_data_config.py
python tools/sonic/smoke_sonic_dataloader.py \
  --config DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml \
  --data-root "$SONIC_DATA_ROOT" \
  --num-workers 0
```

**Submit command:**

```bash
clariden-submit-worktree --rsync -- env \
  CLARIDEN_ENV=dit4dit-sonic-token-actions \
  SONIC_DATA_ROOT=/iopsstor/scratch/cscs/dsimoes/dit4dit/data/sonic-g1-lerobot \
  scripts/clariden/submit_sonic_contract_smoke.sh
```

**Verification:**

Expected log lines:

```text
action shape: [40, 78]
motion_token shape: [40, 64]
left_hand_joints shape: [40, 7]
right_hand_joints shape: [40, 7]
```

If no SONIC dataset exists on Clariden yet, the job must fail early with a clear message naming `$SONIC_DATA_ROOT`, not with a downstream `KeyError` or silent empty dataset.

**Commit:**

```bash
git add scripts/clariden/submit_sonic_contract_smoke.sh docs/clariden_container_runtime.md
git commit -m "test: add Clariden SONIC contract smoke"
```

---

## Task 8: Add model forward/predict smoke in the container

**Objective:** Verify that the container can load the Cosmos-backed DiT4DiT model and produce `[1, 40, 78]` predictions on Clariden.

**Files:**

- Create: `scripts/clariden/submit_sonic_model_smoke.sh`
- Reuse: `tools/sonic/smoke_sonic_model.py` from the SONIC execution plan

**Prerequisites:**

- `$COSMOS_ROOT` exists on Clariden scratch and contains `Cosmos-Predict2.5-2B`.
- `$SONIC_DATA_ROOT` exists and passed the dataloader smoke.
- `dit4dit_sonic_g1.yaml` exists.

**Submit command:**

```bash
clariden-submit-worktree --rsync -- env \
  CLARIDEN_ENV=dit4dit-sonic-token-actions \
  COSMOS_ROOT=/iopsstor/scratch/cscs/dsimoes/dit4dit/models/Cosmos-Predict2.5-2B \
  SONIC_DATA_ROOT=/iopsstor/scratch/cscs/dsimoes/dit4dit/data/sonic-g1-lerobot \
  DIT4DIT_RUN_ROOT=/iopsstor/scratch/cscs/dsimoes/dit4dit/runs/sonic-g1 \
  scripts/clariden/submit_sonic_model_smoke.sh
```

**Job script requirements:**

- one node, one GPU, debug partition if expected to finish under 1 hour
- scratch logs only
- `WANDB_MODE=offline`
- `CUDA_VISIBLE_DEVICES` left to Slurm/container runtime, not hard-coded
- run `tools/sonic/smoke_sonic_model.py`
- assert output shape `[1, 40, 78]`
- write a compact JSON result to `$DIT4DIT_RUN_ROOT/smoke/model-smoke-<jobid>.json`

**Verification:**

- `sacct` completed 0:0
- smoke JSON exists under scratch
- log contains `normalized_actions shape: [1, 40, 78]`

**Commit:**

```bash
git add scripts/clariden/submit_sonic_model_smoke.sh
git commit -m "test: add Clariden SONIC model smoke"
```

---

## Task 9: Add a tiny-overfit Clariden launcher

**Objective:** Verify the container is not just import-correct but trainable on a tiny SONIC subset.

**Files:**

- Create: `scripts/clariden/submit_sonic_tiny_overfit.sh`
- Reuse or create: `examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh`

**Run scope:**

- 10–50 trajectories
- 1 GPU unless memory forces more
- 500 steps initially
- frozen text encoder and VAE
- W&B offline by default
- outputs under `$DIT4DIT_RUN_ROOT/tiny-overfit-<timestamp>`
- logs under `$LOG_ROOT`

**Debug vs normal partition:**

- Use debug only for a ≤1h smoke.
- If the first 100 steps show it will exceed the debug window, stop and submit to a normal partition with an explicit walltime.
- Do not leave debug allocations open after a failed smoke.

**Verification:**

The job passes only if:

- training starts without container/runtime import failures
- at least 100 optimizer steps complete
- total action loss is finite
- per-slice losses are logged for motion token, left hand, and right hand
- checkpoint/log paths are under scratch

**Commit:**

```bash
git add scripts/clariden/submit_sonic_tiny_overfit.sh examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh
git commit -m "train: add Clariden SONIC tiny-overfit launcher"
```

---

## Task 10: Add a verification report template

**Objective:** Make each Clariden container smoke run auditable and comparable.

**Files:**

- Create: `docs/reports/clariden_container_verification_template.md`

**Report fields:**

- repo branch and commit SHA
- image tag and digest
- base image
- platform (`linux/arm64` or `linux/amd64`)
- Clariden EDF path/name
- Slurm job IDs
- scratch roots used
- Python/Torch/CUDA versions
- GPU name
- import smoke result
- config smoke result
- SONIC contract smoke result
- dataloader action shape
- model smoke action shape
- tiny-overfit first/last loss if run
- known caveats and next action

**Verification:**

After the first completed env smoke, fill the report with real evidence and link the log paths. Do not write a victory report from a job that merely submitted. Submission is not evidence; it is a request.

**Commit:**

```bash
git add docs/reports/clariden_container_verification_template.md
git commit -m "docs: add Clariden container verification report template"
```

---

## Final acceptance criteria

The Docker/Clariden verification track is complete when all of the following are true:

1. `docker/clariden/Dockerfile` builds for the Clariden architecture.
2. The image is published by immutable digest.
3. `~/.edf/dit4dit-sonic-token-actions.toml` or equivalent points to that digest and mounts `/app` plus scratch.
4. Clariden env smoke completes on one GPU with CUDA visible.
5. DiT4DiT repo/config smoke passes in the container.
6. SONIC schema/data-config tests pass in the container.
7. Dataloader smoke emits `[40, 78]` from `$SONIC_DATA_ROOT`.
8. Model smoke emits `[1, 40, 78]` using `$COSMOS_ROOT`.
9. Tiny overfit reaches at least 100 finite-loss steps without writing artifacts under home.
10. `docs/reports/clariden_container_verification_template.md` has been filled for the run and cites job IDs/log paths.

---

## Immediate next action

Start with Task 1. The one value that matters most is architecture. If Clariden requires ARM64 and we push an x86 image, the rest of the plan becomes expensive theater.
