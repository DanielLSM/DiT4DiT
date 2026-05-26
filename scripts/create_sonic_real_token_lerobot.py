#!/usr/bin/env python3
"""Create a LeRobot-v2 SONIC G1 dataset with real SONIC encoder tokens.

This converts either:

- GEAR-SONIC `robot_filtered/*.pkl` motions with 29D G1 DoF, or
- BONES compressed CSV trajectories (`*.tar.zst`) with named G1-ish joints,

into a small LeRobot-v2 dataset where:

    observation.state = 29D G1 DoF
    action[0:64]     = real 64D token from model_encoder.onnx
    action[64:71]    = left-hand placeholder/proxy
    action[71:78]    = right-hand placeholder/proxy

The hand channels are zero by default because the public BONES/GEAR motion priors
used here do not contain Dex3 7D hand teleop streams. That makes this a motion-token
pilot dataset, not a full task/hand VLA dataset. The important improvement over the
old fixture is that the first 64 action dimensions are produced by SONIC's real
encoder, not a deterministic proxy.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml
from PIL import Image, ImageDraw

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None

# IsaacLab index subsets used by the SONIC deploy code.
LOWER_BODY_IDX = np.asarray([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int64)
WRIST_IDX = np.asarray([23, 24, 25, 26, 27, 28], dtype=np.int64)

# Observation dimensions from /iopsstor/.../sonic-assets/observation_config.yaml.
ENCODER_DIMS = {
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
}

# 29D IsaacLab-order joint names, inferred from SONIC deploy policy_parameters.hpp.
ISAACLAB_JOINTS = [
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
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _load_nested_joblib(path: Path) -> dict[str, Any]:
    obj = joblib.load(path)
    if isinstance(obj, dict) and len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
        return next(iter(obj.values()))
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected dict in {path}, got {type(obj)!r}")


def _quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def _rot6(q: np.ndarray) -> np.ndarray:
    mat = _quat_to_matrix_xyzw(q)
    return mat[..., :, :2].reshape(*mat.shape[:-2], 6).astype(np.float32)


def _euler_xyz_to_quat(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Euler XYZ radians to xyzw quaternion."""
    cx, sx = np.cos(x / 2), np.sin(x / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    cz, sz = np.cos(z / 2), np.sin(z / 2)
    qw = cx * cy * cz - sx * sy * sz
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    q = np.stack([qx, qy, qz, qw], axis=-1).astype(np.float32)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)


def _window(arr: np.ndarray, start: int, count: int, step: int) -> np.ndarray:
    idx = start + np.arange(count, dtype=np.int64) * step
    idx = np.clip(idx, 0, arr.shape[0] - 1)
    return np.asarray(arr, dtype=np.float32)[idx]


def _vel(x: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(np.asarray(x, dtype=np.float32), axis=0).astype(np.float32) * float(fps)


def _encoder_order(cfg: dict[str, Any]) -> list[str]:
    return [o["name"] for o in cfg["encoder"]["encoder_observations"] if o.get("enabled", True)]


def _build_encoder_obs(order: list[str], dof: np.ndarray, root: np.ndarray, root_quat: np.ndarray, fps: float, start: int) -> np.ndarray:
    dof = np.asarray(dof, dtype=np.float32)
    root = np.asarray(root, dtype=np.float32)
    root_quat = np.asarray(root_quat, dtype=np.float32)
    dof_vel = _vel(dof, fps)
    zeros_smpl = np.zeros((dof.shape[0], 24, 3), dtype=np.float32)
    identity_quat = np.zeros((dof.shape[0], 4), dtype=np.float32)
    identity_quat[:, 3] = 1.0

    vals: dict[str, np.ndarray] = {
        "encoder_mode_4": np.asarray([0, 0, 0, 0], dtype=np.float32),  # g1 mode
        "motion_joint_positions_10frame_step5": _window(dof, start, 10, 5).reshape(-1),
        "motion_joint_velocities_10frame_step5": _window(dof_vel, start, 10, 5).reshape(-1),
        "motion_root_z_position_10frame_step5": _window(root[:, 2:3], start, 10, 5).reshape(-1),
        "motion_root_z_position": root[start : start + 1, 2].reshape(-1),
        "motion_anchor_orientation": _rot6(root_quat[start : start + 1]).reshape(-1),
        "motion_anchor_orientation_10frame_step5": _rot6(_window(root_quat, start, 10, 5)).reshape(-1),
        "motion_joint_positions_lowerbody_10frame_step5": _window(dof[:, LOWER_BODY_IDX], start, 10, 5).reshape(-1),
        "motion_joint_velocities_lowerbody_10frame_step5": _window(dof_vel[:, LOWER_BODY_IDX], start, 10, 5).reshape(-1),
        # G1-only mode does not semantically use these, but the ONNX ABI expects the full union input.
        "vr_3point_local_target": np.zeros((9,), dtype=np.float32),
        "vr_3point_local_orn_target": np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), 3),
        "smpl_joints_10frame_step1": _window(zeros_smpl, start, 10, 1).reshape(-1),
        "smpl_anchor_orientation_10frame_step1": _rot6(_window(identity_quat, start, 10, 1)).reshape(-1),
        "motion_joint_positions_wrists_10frame_step1": _window(dof[:, WRIST_IDX], start, 10, 1).reshape(-1),
    }
    parts = []
    for name in order:
        x = vals[name].astype(np.float32).reshape(-1)
        if x.size != ENCODER_DIMS[name]:
            raise ValueError(f"{name}: expected {ENCODER_DIMS[name]}, got {x.size}")
        parts.append(x)
    return np.concatenate(parts).reshape(1, -1).astype(np.float32)


def _render_frame(dof_t: np.ndarray, root_t: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    img = Image.new("RGB", (width, height), (8, 10, 14))
    draw = ImageDraw.Draw(img)
    cx, cy = width // 2, height // 2
    cx += int(float(np.tanh(root_t[0])) * width * 0.12)
    cy += int(float(np.tanh(root_t[1])) * height * 0.12)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(245, 214, 96))
    vals = np.tanh(np.asarray(dof_t, dtype=np.float32))
    for idx, value in enumerate(vals[:29]):
        angle = 2.0 * math.pi * idx / 29.0
        radius = min(width, height) * (0.12 + 0.25 * (float(value) + 1.0) / 2.0)
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
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if writer.isOpened():
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            return
        writer.release()
    if imageio is None:
        raise RuntimeError("Need cv2 or imageio to write mp4")
    imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)


def _motion_from_robot_pkl(path: Path) -> dict[str, Any]:
    m = _load_nested_joblib(path)
    dof = np.asarray(m["dof"], dtype=np.float32)
    root = np.asarray(m.get("root_trans_offset", np.zeros((dof.shape[0], 3), dtype=np.float32)), dtype=np.float32)
    root_quat = np.asarray(m.get("root_rot", np.tile([0, 0, 0, 1], (dof.shape[0], 1))), dtype=np.float32)
    return {"name": path.stem, "dof": dof, "root": root, "root_quat": root_quat, "fps": int(m.get("fps", 30)), "source": str(path)}


def _list_tar_csvs(tar_path: Path, max_files: int) -> list[str]:
    proc = subprocess.run(["tar", "--zstd", "-tf", str(tar_path)], check=True, capture_output=True, text=True)
    return [line for line in proc.stdout.splitlines() if line.endswith(".csv")][:max_files]


def _motion_from_csv_bytes(name: str, data: bytes, fps: int = 30) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) < 80:
        raise ValueError(f"{name}: too short ({len(rows)} rows)")
    root = np.zeros((len(rows), 3), dtype=np.float32)
    root[:, 0] = [float(r.get("root_translateX", 0.0)) for r in rows]
    root[:, 1] = [float(r.get("root_translateY", 0.0)) for r in rows]
    root[:, 2] = [float(r.get("root_translateZ", 0.0)) for r in rows]
    # BONES CSVs are centimeters-ish for translation; keep pkl-scale meters.
    if np.nanmedian(np.abs(root[:, 2])) > 10.0:
        root = root / 100.0
    rx = np.asarray([float(r.get("root_rotateX", 0.0)) for r in rows], dtype=np.float32)
    ry = np.asarray([float(r.get("root_rotateY", 0.0)) for r in rows], dtype=np.float32)
    rz = np.asarray([float(r.get("root_rotateZ", 0.0)) for r in rows], dtype=np.float32)
    root_quat = _euler_xyz_to_quat(rx, ry, rz)
    dof = np.zeros((len(rows), 29), dtype=np.float32)
    for j, joint in enumerate(ISAACLAB_JOINTS):
        key = f"{joint}_dof"
        if key in rows[0]:
            vals = np.asarray([float(r[key]) for r in rows], dtype=np.float32)
            # CSV joint values are in degrees; GEAR pkl uses radians.
            if np.nanmax(np.abs(vals)) > 2 * math.pi:
                vals = np.deg2rad(vals)
            dof[:, j] = vals
    return {"name": Path(name).stem, "dof": dof, "root": root, "root_quat": root_quat, "fps": fps, "source": name}


def _motions_from_tar(tar_path: Path, max_files: int) -> Iterable[dict[str, Any]]:
    for member in _list_tar_csvs(tar_path, max_files=max_files):
        proc = subprocess.run(["tar", "--zstd", "-xOf", str(tar_path), member], check=True, capture_output=True)
        yield _motion_from_csv_bytes(member, proc.stdout)


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


def build_dataset(
    assets: Path,
    output_root: Path,
    dataset_name: str,
    robot_pkls: list[Path],
    csv_files: list[Path],
    tar_zst: list[Path],
    max_csv_files_per_tar: int,
    episode_len: int,
    stride: int,
    max_episodes: int,
    image_size: tuple[int, int],
) -> Path:
    cfg = yaml.safe_load((assets / "observation_config.yaml").read_text())
    order = _encoder_order(cfg)
    expected_dim = sum(ENCODER_DIMS[n] for n in order)
    sess = ort.InferenceSession(str(assets / "model_encoder.onnx"), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    if inp.name != "obs_dict" or list(inp.shape) != [1, expected_dim] or list(out.shape) != [1, 64]:
        raise RuntimeError(f"Unexpected encoder ABI: {inp.name} {inp.shape} -> {out.name} {out.shape}; expected [1,{expected_dim}] -> [1,64]")

    motions: list[dict[str, Any]] = []
    motions.extend(_motion_from_robot_pkl(p) for p in robot_pkls)
    for csv_path in csv_files:
        motions.append(_motion_from_csv_bytes(str(csv_path), csv_path.read_bytes()))
    for tar_path in tar_zst:
        motions.extend(_motions_from_tar(tar_path, max_files=max_csv_files_per_tar))
    if not motions:
        raise ValueError("No source motions provided")

    dataset_dir = output_root / dataset_name
    data_dir = dataset_dir / "data" / "chunk-000"
    video_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.ego_view"
    meta_dir = dataset_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta: list[dict[str, Any]] = []
    task_texts: dict[int, str] = {}
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    global_index = 0
    ep_idx = 0

    for motion in motions:
        dof = np.asarray(motion["dof"], dtype=np.float32)
        root = np.asarray(motion["root"], dtype=np.float32)
        root_quat = np.asarray(motion["root_quat"], dtype=np.float32)
        fps = int(motion.get("fps", 30))
        if dof.ndim != 2 or dof.shape[1] != 29 or dof.shape[0] < episode_len + 50:
            continue
        starts = list(range(0, dof.shape[0] - episode_len - 50 + 1, stride))
        for start in starts:
            if ep_idx >= max_episodes:
                break
            rows = []
            ep_dof = dof[start : start + episode_len].astype(np.float32)
            ep_root = root[start : start + episode_len].astype(np.float32)
            ep_actions = np.zeros((episode_len, 78), dtype=np.float32)
            for t in range(episode_len):
                x = _build_encoder_obs(order, dof, root, root_quat, float(fps), start + t)
                token = sess.run([out.name], {inp.name: x})[0].astype(np.float32).reshape(64)
                # Hand fields are absent in the source motion prior. Preserve ABI with zeros.
                action = np.concatenate([token, np.zeros(14, dtype=np.float32)], axis=0).astype(np.float32)
                ep_actions[t] = action
                rows.append(
                    {
                        "observation.state": ep_dof[t],
                        "action": action,
                        "timestamp": np.float32(t / fps),
                        "frame_index": t,
                        "episode_index": ep_idx,
                        "index": global_index,
                        "task_index": ep_idx,
                    }
                )
                global_index += 1
            pd.DataFrame(rows).to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet", index=False)
            frames = np.stack([_render_frame(ep_dof[i], ep_root[i], image_size) for i in range(episode_len)], axis=0)
            _write_video(video_dir / f"episode_{ep_idx:06d}.mp4", frames, fps=fps)
            task_texts[ep_idx] = f"imitate SONIC G1 motion-token trajectory from {motion['name']}"
            episodes_meta.append(
                {
                    "episode_index": ep_idx,
                    "tasks": [ep_idx],
                    "length": episode_len,
                    "dataset_source": motion["source"],
                    "source_start_index": start,
                    "action_semantics": "64D real SONIC model_encoder token + 14D zero hand placeholders",
                }
            )
            all_states.append(ep_dof)
            all_actions.append(ep_actions)
            ep_idx += 1
        if ep_idx >= max_episodes:
            break

    if not episodes_meta:
        raise RuntimeError("No episodes generated")

    state_all = np.concatenate(all_states, axis=0)
    action_all = np.concatenate(all_actions, axis=0)
    width, height = image_size
    info = {
        "codebase_version": "v2.0",
        "robot_type": "sonic_g1_78d",
        "fps": 30,
        "video": True,
        "total_episodes": len(episodes_meta),
        "total_frames": int(sum(ep["length"] for ep in episodes_meta)),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.ego_view": {"dtype": "video", "shape": [height, width, 3], "names": ["height", "width", "channel"], "video_info": {"video.fps": 30, "video.codec": "mp4v"}},
            "observation.state": {"dtype": "float32", "shape": [29], "names": ["g1_dof"]},
            "action": {"dtype": "float32", "shape": [78], "names": ["sonic_action_real_token_zero_hands"]},
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
    meta_dir.joinpath("info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    meta_dir.joinpath("modality.json").write_text(json.dumps(modality, indent=2, sort_keys=True) + "\n")
    meta_dir.joinpath("episodes.jsonl").write_text("".join(json.dumps(ep, sort_keys=True) + "\n" for ep in episodes_meta))
    meta_dir.joinpath("tasks.jsonl").write_text("".join(json.dumps({"task_index": k, "task": v}, sort_keys=True) + "\n" for k, v in sorted(task_texts.items())))
    meta_dir.joinpath("stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    meta_dir.joinpath("stats_gr00t.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    summary = {
        "dataset_dir": str(dataset_dir),
        "episodes": len(episodes_meta),
        "frames": int(info["total_frames"]),
        "action_shape": [78],
        "token_stats": {"finite": bool(np.isfinite(action_all[:, :64]).all()), "std": float(np.std(action_all[:, :64])), "abs_max": float(np.max(np.abs(action_all[:, :64])))},
        "hand_stats": {"finite": bool(np.isfinite(action_all[:, 64:]).all()), "std": float(np.std(action_all[:, 64:])), "abs_max": float(np.max(np.abs(action_all[:, 64:])))},
        "sources_used": [ep["dataset_source"] for ep in episodes_meta[:10]],
    }
    dataset_dir.joinpath("conversion_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return dataset_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dataset-name", default="sonic_g1_real_token_lerobot")
    p.add_argument("--robot-pkl", type=Path, action="append", default=[])
    p.add_argument("--csv-file", type=Path, action="append", default=[])
    p.add_argument("--csv-dir", type=Path, action="append", default=[])
    p.add_argument("--tar-zst", type=Path, action="append", default=[])
    p.add_argument("--max-csv-files-per-tar", type=int, default=8)
    p.add_argument("--episode-len", type=int, default=64)
    p.add_argument("--stride", type=int, default=96)
    p.add_argument("--max-episodes", type=int, default=24)
    p.add_argument("--image-width", type=int, default=96)
    p.add_argument("--image-height", type=int, default=96)
    args = p.parse_args()
    csv_files = list(args.csv_file)
    for csv_dir in args.csv_dir:
        csv_files.extend(sorted(csv_dir.rglob("*.csv")))
    build_dataset(
        assets=args.assets,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        robot_pkls=args.robot_pkl,
        csv_files=csv_files,
        tar_zst=args.tar_zst,
        max_csv_files_per_tar=args.max_csv_files_per_tar,
        episode_len=args.episode_len,
        stride=args.stride,
        max_episodes=args.max_episodes,
        image_size=(args.image_width, args.image_height),
    )


if __name__ == "__main__":
    main()
