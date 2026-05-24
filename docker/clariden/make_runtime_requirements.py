#!/usr/bin/env python3
"""Create a container-runtime requirements file for the Clariden image.

The NVIDIA PyTorch base image already owns torch and CUDA wheel packages. This
script removes those entries from the repository lock-style requirements so the
container build does not replace the base image's CUDA stack with pip wheels.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Packages supplied by the NVIDIA PyTorch base image or tightly coupled to its
# CUDA runtime. Keep names canonicalized to PEP 503 form: lowercase with '-'.
DROP_EXACT = {
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    # Not imported by the repo; remove it so broad smoke checks for torch/CUDA
    # wheel leakage stay clean while the base image supplies real torch.
    "torch-einops-utils",
    # PyPI only publishes x86_64 Linux wheels for these pinned releases. The
    # Clariden image is linux/arm64, and the repo has PyAV/torchvision fallback
    # video backends for the first smoke tests.
    "decord",
    "eva-decord",
}
DROP_NVIDIA_CUDA_RE = re.compile(r"^nvidia-[a-z0-9-]+-cu\d+$")

NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(?:\[.*?\])?\s*(?:==|~=|!=|<=|>=|<|>|===|@|;|$)")


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith(("-r", "--requirement", "-c", "--constraint")):
        return None
    match = NAME_RE.match(stripped)
    if not match:
        return None
    return canonicalize_name(match.group(1))


def should_drop(line: str, extra_excludes: set[str]) -> tuple[bool, str | None]:
    name = requirement_name(line)
    if name is None:
        return False, None
    if name in DROP_EXACT or name in extra_excludes:
        return True, name
    if DROP_NVIDIA_CUDA_RE.match(name):
        return True, name
    return False, name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source requirements.txt")
    parser.add_argument("--output", required=True, type=Path, help="Sanitized output path")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional package name to remove. May be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra_excludes = {canonicalize_name(name) for name in args.exclude}

    kept: list[str] = []
    dropped: list[str] = []
    for line in args.input.read_text().splitlines():
        drop, name = should_drop(line, extra_excludes)
        if drop:
            dropped.append(name or line.strip())
            continue
        kept.append(line.rstrip())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept).rstrip() + "\n")

    print(
        f"wrote {args.output} with {len(kept)} lines; "
        f"dropped {len(dropped)} CUDA/base-image entries: {', '.join(dropped)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
