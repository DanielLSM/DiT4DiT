# Clariden Container Runtime Inventory

This inventory pins the Clariden/Sarus runtime facts used by the DiT4DiT SONIC branch container plan. It was collected from the `experiment/sonic-token-actions` worktree before writing the Dockerfile/EDF bridge.

## Login-node facts

Collected with:

```bash
ssh clariden 'set -e; uname -a; uname -m; command -v sarus || true; command -v enroot || true; command -v sqshfs || true; ls -la ~/.edf 2>/dev/null || true'
```

- Login host observed: `clariden-ln004`
- Login kernel/OS string: `Linux clariden-ln004 6.4.0-150600.23.47_15.0.10-cray_shasta_c_64k ... aarch64 GNU/Linux`
- Login architecture: `aarch64`
- `sarus`: not present in `PATH` on the login node
- `enroot`: `/usr/bin/enroot`
- `sqshfs`: not present in `PATH` on the login node
- EDF directory: `/users/dsimoes/.edf`

## Debug/GPU-node facts

Collected with a short debug allocation:

```bash
ssh clariden 'srun --account=a143 --partition=debug --nodes=1 --ntasks=1 --time=00:05:00 bash -lc '\''echo GPU_NODE_HOST=$(hostname); echo GPU_NODE_UNAME=$(uname -a); echo GPU_NODE_ARCH=$(uname -m); nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvidia-smi | sed -n "1,12p"'\'''
```

- Debug host observed: `nid007665`
- GPU-node kernel/OS string: `Linux nid007665 6.4.0-150600.23.47_15.0.10-cray_shasta_c_64k ... aarch64 GNU/Linux`
- GPU-node architecture: `aarch64`
- GPU model: `NVIDIA GH200 120GB`
- GPUs per debug node: `4`
- Driver: `590.48.01`
- CUDA reported by `nvidia-smi`: `13.1`
- Debug partition: `debug`
- Debug partition walltime limit observed by `sinfo`: `1:30:00`
- Slurm account: `a143`

## Container runtime and EDF schema

Existing EDF examples under `/users/dsimoes/.edf` use a compact TOML schema consumed by Slurm's `--environment=<name>` support:

```toml
image = "/capstor/scratch/cscs/dsimoes/images/example.sqsh"

mounts = [
  "/users/dsimoes/worktrees/<repo>/<branch>:/app",
  "/capstor",
  "/iopsstor",
  "/users"
]

workdir = "/app"

[env]
PYTHONPATH = "/app"
HF_HOME = "/iopsstor/scratch/cscs/dsimoes/dit4dit/cache/huggingface"
TORCH_HOME = "/iopsstor/scratch/cscs/dsimoes/dit4dit/cache/torch"
WANDB_DIR = "/iopsstor/scratch/cscs/dsimoes/dit4dit/wandb"
```

Relevant keys:

- Image URI/path key: top-level `image`
- Bind mounts key: top-level `mounts`, as an array of strings
- Container working directory key: top-level `workdir`
- Environment variables: `[env]` table
- Existing worktree mount convention: branch worktrees live under `/users/dsimoes/worktrees/<repo>/<branch>` and are mounted to `/app`

Existing examples commonly mount `/capstor`, `/iopsstor`, and `/users` wholesale so container-visible paths match host paths. For this DiT4DiT branch, the minimum intended mount is:

```toml
mounts = [
  "/users/dsimoes/worktrees/DiT4DiT/experiment/sonic-token-actions:/app",
  "/iopsstor/scratch/cscs/dsimoes/dit4dit:/iopsstor/scratch/cscs/dsimoes/dit4dit",
  "/capstor",
  "/users"
]
```

Mounting all of `/iopsstor` is acceptable if the EDF follows the existing local convention, but the DiT4DiT scripts must still write generated datasets, caches, checkpoints, logs, W&B files, and run outputs under `/iopsstor/scratch/cscs/dsimoes/dit4dit`.

## Image format and architecture decision

- Clariden login and GPU nodes are `aarch64`.
- Build the Docker image for `linux/arm64`.
- The EDF examples point at local `.sqsh` images rather than Docker registry URLs.
- The plan can still publish a registry image, but Clariden execution should expect an imported/enroot-compatible squashfs image path unless a direct registry pull is explicitly verified.

## Registry authentication

No registry pull authentication was verified during inventory. Existing EDFs use local `.sqsh` image files under `/capstor/...`, so the safe first path is:

1. Build/push the `linux/arm64` image outside Clariden.
2. Import or convert it to a `.sqsh` image under a scratch/store path visible to Clariden.
3. Reference that `.sqsh` path from `~/.edf/dit4dit-sonic-token-actions.toml`.

Do not put registry tokens, Hugging Face tokens, W&B keys, SSH keys, robot IPs, or robot credentials in the Dockerfile, image, EDF, Slurm logs, or GitHub Actions logs.
