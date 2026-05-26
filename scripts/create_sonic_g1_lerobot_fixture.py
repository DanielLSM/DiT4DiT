#!/usr/bin/env python3
"""Create a tiny LeRobot-v2 style SONIC/Unitree-G1 smoke fixture.

The input is the small `nvidia/GEAR-SONIC --sample` robot_filtered joblib pickle.
It contains real Unitree G1 retargeted motion (`dof`, root pose, fps) but no camera
stream and no exported VLA motion tokens. This script therefore preserves real G1
state trajectories and derives deterministic proxy 78D actions for structural
DiT4DiT/SONIC smoke tests:

    action[0:64]  = smooth proxy motion-token features from G1 dof/root motion
    action[64:71] = left-hand proxy joints from G1 dof slices
    action[71:78] = right-hand proxy joints from G1 dof slices

This fixture is intentionally small and is not a scientific dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - fallback covered in runtime use if cv2 is absent
    cv2 = None

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None


def _load_joblib(path: Path) -> dict[str, Any]:
    try:
        import joblib  # type: ignore
    except Exception as exc:  # pragma: no cover - tested by command line in envs with/without joblib
        raise SystemExit(
            "Missing dependency `joblib`, required for compressed GEAR-SONIC .pkl files. "
            "Install it in the active environment or run inside the DiT4DiT Clariden container."
        ) from exc
    loaded = joblib.load(path)
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f"Expected non-empty dict in {path}, got {type(loaded)!r}")
    # GEAR sample shape: {"walk_forward...": {...}}
    first = next(iter(loaded.values()))
    if not isinstance(first, dict):
        raise ValueError(f"Expected nested motion dict in {path}")
    return first


def _stats(array: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(array, dtype=np.float32)
    return {
        "mean": np.mean(x, axis=0).astype(float).tolist(),
        "std": np.std(x, axis=0).astype(float).tolist(),
        "min": np.min(x, axis=0).astype(float).tolist(),
        "max": np.max(x, axis=0).astype(float).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).astype(float).tolist(),
    }


def _normalize_columns(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def _repeat_to_dim(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[1] >= dim:
        return x[:, :dim]
    reps = math.ceil(dim / x.shape[1])
    return np.tile(x, (1, reps))[:, :dim]


def make_proxy_actions(dof: np.ndarray, root_trans: np.ndarray | None) -> np.ndarray:
    """Build deterministic 78D proxy actions from real G1 trajectory arrays."""
    dof = np.asarray(dof, dtype=np.float32)
    velocity = np.gradient(dof, axis=0).astype(np.float32)
    pieces = [_normalize_columns(dof), _normalize_columns(velocity)]
    if root_trans is not None:
        root_trans = np.asarray(root_trans, dtype=np.float32)
        root_vel = np.gradient(root_trans, axis=0).astype(np.float32)
        pieces.extend([_normalize_columns(root_trans), _normalize_columns(root_vel)])
    features = np.concatenate(pieces, axis=1)
    motion_token_proxy = np.tanh(_repeat_to_dim(features, 64)).astype(np.float32)

    # The public sample has G1 body dof but no real hand teleop stream. Use stable,
    # bounded slices as structural placeholders so the 64+7+7 SONIC ABI is exercised.
    left_src = dof[:, :7] if dof.shape[1] >= 7 else _repeat_to_dim(dof, 7)
    right_src = dof[:, 7:14] if dof.shape[1] >= 14 else _repeat_to_dim(dof, 7)
    left_hand = np.tanh(_normalize_columns(left_src)).astype(np.float32)
    right_hand = np.tanh(_normalize_columns(right_src)).astype(np.float32)
    return np.concatenate([motion_token_proxy, left_hand, right_hand], axis=1).astype(np.float32)


def _frame_from_state(dof_t: np.ndarray, root_t: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    """Render a tiny deterministic diagnostic frame from the G1 state."""
    width, height = size
    img = Image.new("RGB", (width, height), (8, 10, 14))
    draw = ImageDraw.Draw(img)
    center_x = width // 2
    center_y = height // 2
    scale = min(width, height) * 0.30
    vals = np.tanh(dof_t.astype(np.float32))
    if root_t is not None:
        offset_x = int(float(np.tanh(root_t[0])) * width * 0.12)
        offset_y = int(float(np.tanh(root_t[1])) * height * 0.12)
    else:
        offset_x = offset_y = 0
    cx, cy = center_x + offset_x, center_y + offset_y
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(245, 214, 96))
    for idx, value in enumerate(vals[:24]):
        angle = (2.0 * math.pi * idx) / 24.0
        radius = scale * (0.45 + 0.45 * (float(value) + 1.0) / 2.0)
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        color = (70 + (idx * 31) % 150, 110 + (idx * 17) % 120, 180 + (idx * 11) % 70)
        draw.line((cx, cy, x, y), fill=color, width=2)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    return np.asarray(img, dtype=np.uint8)


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        h, w = frames.shape[1:3]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        if writer.isOpened():
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            return
        writer.release()
    if imageio is None:
        raise RuntimeError("Neither cv2 VideoWriter nor imageio is available to write mp4 video")
    imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def build_fixture(
    source_pkl: Path,
    output_root: Path,
    dataset_name: str,
    episode_len: int,
    stride: int,
    max_episodes: int,
    image_size: tuple[int, int],
) -> Path:
    motion = _load_joblib(source_pkl)
    dof = np.asarray(motion["dof"], dtype=np.float32)
    root_trans = np.asarray(motion.get("root_trans_offset"), dtype=np.float32) if "root_trans_offset" in motion else None
    fps = int(motion.get("fps", 30))
    if dof.ndim != 2 or dof.shape[1] != 29:
        raise ValueError(f"Expected G1 dof shape [T,29], got {dof.shape}")
    if dof.shape[0] < episode_len:
        raise ValueError(f"Need at least {episode_len} frames, got {dof.shape[0]}")

    actions = make_proxy_actions(dof, root_trans)
    dataset_dir = output_root / dataset_name
    data_dir = dataset_dir / "data" / "chunk-000"
    video_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.ego_view"
    meta_dir = dataset_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    starts = list(range(0, dof.shape[0] - episode_len + 1, stride))[:max_episodes]
    if not starts:
        starts = [0]

    episodes_meta: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    global_index = 0
    task_index = 0
    task_text = "walk forward with Unitree G1 using SONIC motion-token action ABI"

    for episode_index, start in enumerate(starts):
        end = start + episode_len
        ep_dof = dof[start:end].astype(np.float32)
        ep_root = root_trans[start:end].astype(np.float32) if root_trans is not None else None
        ep_actions = actions[start:end].astype(np.float32)
        rows = []
        for local_idx in range(episode_len):
            rows.append(
                {
                    "observation.state": ep_dof[local_idx],
                    "action": ep_actions[local_idx],
                    "timestamp": np.float32(local_idx / fps),
                    "frame_index": local_idx,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": task_index,
                }
            )
            global_index += 1
        pd.DataFrame(rows).to_parquet(data_dir / f"episode_{episode_index:06d}.parquet", index=False)
        frames = np.stack(
            [_frame_from_state(ep_dof[i], ep_root[i] if ep_root is not None else None, image_size) for i in range(episode_len)],
            axis=0,
        )
        _write_video(video_dir / f"episode_{episode_index:06d}.mp4", frames, fps=fps)
        episodes_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [task_index],
                "length": episode_len,
                "dataset_source": "nvidia/GEAR-SONIC sample_data/robot_filtered",
                "source_pkl": source_pkl.name,
                "source_start_index": start,
            }
        )
        all_states.append(ep_dof)
        all_actions.append(ep_actions)

    state_all = np.concatenate(all_states, axis=0)
    action_all = np.concatenate(all_actions, axis=0)
    width, height = image_size
    info = {
        "codebase_version": "v2.0",
        "robot_type": "sonic_g1_78d",
        "fps": fps,
        "video": True,
        "total_episodes": len(episodes_meta),
        "total_frames": int(sum(ep["length"] for ep in episodes_meta)),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": fps, "video.codec": "mp4v"},
            },
            "observation.state": {"dtype": "float32", "shape": [29], "names": ["g1_dof"]},
            "action": {"dtype": "float32", "shape": [78], "names": ["sonic_action"]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    modality = {
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "state": {
            "g1_dof": {
                "start": 0,
                "end": 29,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.state",
            }
        },
        "action": {
            "motion_token": {
                "start": 0,
                "end": 64,
                "rotation_type": None,
                "absolute": False,
                "dtype": "float32",
                "original_key": "action",
            },
            "left_hand_joints": {
                "start": 64,
                "end": 71,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "original_key": "action",
            },
            "right_hand_joints": {
                "start": 71,
                "end": 78,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "original_key": "action",
            },
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    stats = {"observation.state": _stats(state_all), "action": _stats(action_all)}

    (meta_dir / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (meta_dir / "stats_gr00t.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": task_index, "task": task_text}])

    manifest = {
        "dataset_dir": str(dataset_dir),
        "source_pkl": str(source_pkl),
        "episode_len": episode_len,
        "stride": stride,
        "num_episodes": len(episodes_meta),
        "state_shape": list(state_all.shape),
        "action_shape": list(action_all.shape),
        "action_layout": "64 motion_token_proxy + 7 left_hand_proxy + 7 right_hand_proxy",
        "fps": fps,
        "image_size_wh": [width, height],
    }
    (meta_dir / "sonic_g1_fixture_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pkl", type=Path, required=True, help="GEAR-SONIC robot_filtered .pkl path")
    parser.add_argument("--output-root", type=Path, required=True, help="Parent directory for the LeRobot dataset")
    parser.add_argument("--dataset-name", default="sonic_g1_sample_lerobot")
    parser.add_argument("--episode-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument("--max-episodes", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--image-height", type=int, default=96)
    args = parser.parse_args()

    dataset_dir = build_fixture(
        source_pkl=args.source_pkl,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        episode_len=args.episode_len,
        stride=args.stride,
        max_episodes=args.max_episodes,
        image_size=(args.image_width, args.image_height),
    )
    print(json.dumps({"dataset_dir": str(dataset_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
