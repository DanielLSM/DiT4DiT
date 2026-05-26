#!/usr/bin/env python3
"""Convert public SONIC Unitree-G1 LeRobot datasets to DiT4DiT's 29D/78D ABI.

Public datasets such as `theconstruct-ai/push_box_gear_sonic` store SONIC VLA
frames as:

    observation.state      [43]  Unitree G1 WBC joints, including hands
    action.wbc             [43]  Unitree G1 WBC target joints, including hands
    action.motion_token    [64]  SONIC latent motion token

The DiT4DiT branch uses the narrower policy ABI already validated against SONIC:

    observation.state      [29]  G1 body/arm joints, excluding hand joints
    action                 [78]  64D motion token + 7D left hand + 7D right hand

This script materializes that layout as a LeRobot-v2 style directory under scratch.
It deliberately validates the Unitree G1 joint names rather than trusting the source
`robot_type` field, because that metadata is null in the public datasets. Metadata
lies; joint names leave fingerprints.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

EXPECTED_G1_WBC_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

BODY29_IDX = np.asarray(list(range(22)) + list(range(29, 36)), dtype=np.int64)
LEFT_HAND_SLICE_43 = slice(22, 29)
RIGHT_HAND_SLICE_43 = slice(36, 43)


def _shape_tuple(feature: dict[str, Any], key: str) -> tuple[int, ...]:
    shape = feature.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"{key} missing list shape")
    return tuple(int(x) for x in shape)


def validate_unitree_g1_sonic_features(features: dict[str, Any]) -> None:
    required_shapes = {
        "observation.images.ego_view": (480, 640, 3),
        "observation.state": (43,),
        "action.wbc": (43,),
        "action.motion_token": (64,),
    }
    for key, shape in required_shapes.items():
        if key not in features:
            raise ValueError(f"missing required feature {key}")
        if _shape_tuple(features[key], key) != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {_shape_tuple(features[key], key)}")
    for key in ("observation.state", "action.wbc"):
        names = features[key].get("names")
        if names != EXPECTED_G1_WBC_NAMES:
            raise ValueError(f"{key}: unexpected Unitree G1 joint names")


def _as_vector(x: Any, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape != (dim,):
        raise ValueError(f"{name}: expected shape ({dim},), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: non-finite values")
    return arr


def build_dit4dit_state(state43: Any) -> np.ndarray:
    state = _as_vector(state43, 43, "observation.state")
    return state[BODY29_IDX].astype(np.float32, copy=False)


def build_dit4dit_action(motion_token64: Any, wbc43: Any) -> np.ndarray:
    token = _as_vector(motion_token64, 64, "action.motion_token")
    wbc = _as_vector(wbc43, 43, "action.wbc")
    return np.concatenate([token, wbc[LEFT_HAND_SLICE_43], wbc[RIGHT_HAND_SLICE_43]], axis=0).astype(np.float32, copy=False)


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _download_if_needed(hf_repo: str | None, download_dir: Path | None, source_dir: Path | None) -> Path:
    if source_dir is not None:
        return source_dir
    if not hf_repo:
        raise ValueError("provide either --source-dir or --hf-repo")
    if download_dir is None:
        raise ValueError("--download-dir is required with --hf-repo")
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=hf_repo, repo_type="dataset", local_dir=str(download_dir), local_dir_use_symlinks=False)
    return Path(local)


def _copy_videos(source_dir: Path, output_dir: Path) -> None:
    src = source_dir / "videos"
    if not src.exists():
        raise FileNotFoundError(f"source has no videos directory: {src}")
    dst = output_dir / "videos"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def convert_dataset(source_dir: Path, output_root: Path, dataset_name: str | None = None, max_episodes: int | None = None) -> Path:
    source_dir = source_dir.resolve()
    info_path = source_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing source meta/info.json: {info_path}")
    source_info = json.loads(info_path.read_text(encoding="utf-8"))
    features = source_info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{info_path} has no features dict")
    validate_unitree_g1_sonic_features(features)

    dataset_name = dataset_name or f"{source_dir.name}_dit4dit_sonic_lerobot"
    output_dir = output_root / dataset_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    source_parquets = sorted((source_dir / "data").glob("chunk-*/episode_*.parquet"))
    if max_episodes is not None:
        source_parquets = source_parquets[:max_episodes]
    if not source_parquets:
        raise RuntimeError(f"no source parquet files found under {source_dir / 'data'}")

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episodes_meta: list[dict[str, Any]] = []
    global_index = 0
    for out_ep_idx, pq in enumerate(source_parquets):
        df = pd.read_parquet(pq)
        required_cols = ["observation.state", "action.wbc", "action.motion_token"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{pq}: missing columns {missing}")
        states = np.stack([build_dit4dit_state(v) for v in df["observation.state"].to_list()])
        actions = np.stack([build_dit4dit_action(tok, wbc) for tok, wbc in zip(df["action.motion_token"].to_list(), df["action.wbc"].to_list())])
        n = len(df)
        if n < 40:
            raise ValueError(f"{pq}: only {n} frames; need at least 40 for SONIC horizon")
        out_df = pd.DataFrame(
            {
                "observation.state": list(states.astype(np.float32)),
                "action": list(actions.astype(np.float32)),
                "timestamp": df["timestamp"].to_numpy(np.float32) if "timestamp" in df else np.arange(n, dtype=np.float32) / float(source_info.get("fps", 50)),
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, out_ep_idx, dtype=np.int64),
                "index": np.arange(global_index, global_index + n, dtype=np.int64),
                "task_index": df["task_index"].to_numpy(np.int64) if "task_index" in df else np.zeros(n, dtype=np.int64),
            }
        )
        out_df.to_parquet(output_dir / "data" / "chunk-000" / f"episode_{out_ep_idx:06d}.parquet", index=False)
        episodes_meta.append(
            {
                "episode_index": out_ep_idx,
                "length": int(n),
                "source_dataset": str(source_dir),
                "source_parquet": str(pq.relative_to(source_dir)),
                "action_semantics": "64D SONIC motion_token + 7D left hand + 7D right hand from action.wbc",
                "tasks": sorted(set(int(x) for x in out_df["task_index"].to_list())),
            }
        )
        all_states.append(states)
        all_actions.append(actions)
        global_index += n

    _copy_videos(source_dir, output_dir)
    source_tasks = _read_jsonl(source_dir / "meta" / "tasks.jsonl")
    if not source_tasks:
        source_tasks = [{"task_index": 0, "task": "SONIC Unitree G1 task"}]

    fps = float(source_info.get("fps", 50))
    video_shape = features["observation.images.ego_view"]["shape"]
    video_info = features["observation.images.ego_view"].get("info", {}) or {}
    info = {
        "codebase_version": "v2.0",
        "robot_type": "sonic_g1_78d",
        "fps": fps,
        "video": True,
        "total_episodes": len(episodes_meta),
        "total_frames": int(global_index),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": video_shape,
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": fps, "video.codec": video_info.get("video.codec", "h264")},
            },
            "observation.state": {"dtype": "float32", "shape": [29], "names": ["g1_body_no_hands"]},
            "action": {"dtype": "float32", "shape": [78], "names": ["sonic_motion_token_plus_hands"]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
        "source_hf_dataset": source_info.get("repo_id"),
        "source_dir": str(source_dir),
    }
    modality = {
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "state": {"g1_dof": {"start": 0, "end": 29, "rotation_type": None, "absolute": True, "dtype": "float32", "original_key": "observation.state"}},
        "action": {
            "motion_token": {"start": 0, "end": 64, "rotation_type": None, "absolute": False, "dtype": "float32", "original_key": "action"},
            "left_hand_joints": {"start": 64, "end": 71, "rotation_type": None, "absolute": True, "dtype": "float32", "original_key": "action"},
            "right_hand_joints": {"start": 71, "end": 78, "rotation_type": None, "absolute": True, "dtype": "float32", "original_key": "action"},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    state_all = np.concatenate(all_states, axis=0)
    action_all = np.concatenate(all_actions, axis=0)
    stats = {"observation.state": _stats(state_all), "action": _stats(action_all)}
    (output_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "meta" / "modality.json").write_text(json.dumps(modality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "meta" / "episodes.jsonl", episodes_meta)
    _write_jsonl(output_dir / "meta" / "tasks.jsonl", source_tasks)
    (output_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "meta" / "stats_gr00t.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "ok": True,
        "source_dir": str(source_dir),
        "dataset_dir": str(output_dir),
        "episodes": len(episodes_meta),
        "frames": int(global_index),
        "fps": fps,
        "state_shape": [29],
        "action_shape": [78],
        "motion_token_std": float(np.std(action_all[:, :64])),
        "left_hand_std": float(np.std(action_all[:, 64:71])),
        "right_hand_std": float(np.std(action_all[:, 71:78])),
        "tasks": source_tasks,
    }
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-repo", default=None, help="Optional HF dataset repo id, e.g. theconstruct-ai/gear_sonic_test")
    p.add_argument("--download-dir", type=Path, default=None, help="Scratch path for HF dataset download")
    p.add_argument("--source-dir", type=Path, default=None, help="Existing local/source LeRobot dataset directory")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()
    source_dir = _download_if_needed(args.hf_repo, args.download_dir, args.source_dir)
    convert_dataset(source_dir=source_dir, output_root=args.output_root, dataset_name=args.dataset_name, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
