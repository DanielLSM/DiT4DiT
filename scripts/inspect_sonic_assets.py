#!/usr/bin/env python3
"""Inspect staged GEAR-SONIC assets for the DiT4DiT semantic-token gate."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml


def describe_obj(obj, depth=0, max_depth=4):
    indent = "  " * depth
    if depth > max_depth:
        return f"{indent}..."
    if isinstance(obj, dict):
        lines = [f"{indent}dict[{len(obj)}]"]
        for k in list(obj.keys())[:20]:
            lines.append(f"{indent}  {k!r}: {describe_obj(obj[k], depth + 2, max_depth).lstrip()}")
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        lines = [f"{indent}{type(obj).__name__}[{len(obj)}]"]
        for i, v in enumerate(obj[:5]):
            lines.append(f"{indent}  [{i}]: {describe_obj(v, depth + 2, max_depth).lstrip()}")
        return "\n".join(lines)
    if isinstance(obj, np.ndarray):
        summary = f"ndarray shape={obj.shape} dtype={obj.dtype}"
        if obj.size and np.issubdtype(obj.dtype, np.number):
            arr = obj.astype(np.float64, copy=False)
            summary += f" finite={np.isfinite(arr).all()} min={np.nanmin(arr):.6g} max={np.nanmax(arr):.6g} mean={np.nanmean(arr):.6g}"
        return indent + summary
    return indent + repr(obj)[:200]


def load_pkl(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    assets = args.assets

    print("ASSETS", assets)
    print("FILES")
    for p in sorted(assets.rglob("*")):
        if p.is_file() or p.is_symlink():
            print(f"  {p.relative_to(assets)}\t{p.stat().st_size}")

    cfg_path = assets / "observation_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    print("\nOBS_CONFIG")
    print(json.dumps(cfg, indent=2)[:8000])

    print("\nONNX_RUNTIME")
    try:
        import onnxruntime as ort
        print("onnxruntime", ort.__version__, "providers", ort.get_available_providers())
        for name in ["model_encoder.onnx", "model_decoder.onnx"]:
            sess = ort.InferenceSession(str(assets / name), providers=["CPUExecutionProvider"])
            print("MODEL", name)
            for x in sess.get_inputs():
                print("  input", x.name, x.shape, x.type)
            for y in sess.get_outputs():
                print("  output", y.name, y.shape, y.type)
    except Exception as e:
        print("NO_OR_FAILED_ONNXRUNTIME", repr(e))

    print("\nSAMPLE_PKLS")
    for p in sorted((assets / "sample_data").rglob("*.pkl")):
        print("---", p.relative_to(assets))
        obj = load_pkl(p)
        print(describe_obj(obj))


if __name__ == "__main__":
    main()
