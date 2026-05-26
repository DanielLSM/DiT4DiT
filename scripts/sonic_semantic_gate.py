#!/usr/bin/env python3
"""Minimal semantic gate for staged GEAR-SONIC token assets.

The gate intentionally stays small and evidence-driven:

1. Validate staged ONNX/config/sample schemas.
2. Build real, finite encoder observations from the public robot/smpl sample
   trajectories, in the exact observation_config.yaml order.
3. Run the encoder for g1/teleop/smpl modes and reject dead/constant tokens.
4. Feed tokens plus real G1 history observations through the decoder and reject
   non-finite or pathological actions.

This is not a policy-quality benchmark. It is a semantic ABI smoke gate: do the
staged assets produce non-degenerate motion tokens/actions from real motion data?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import yaml

LOWER_BODY_IDX = np.asarray([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int64)
WRIST_IDX = np.asarray([23, 24, 25, 26, 27, 28], dtype=np.int64)
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
POLICY_DIMS = {
    "token_state": 64,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
}


def _nested_motion(path: Path) -> dict[str, Any]:
    obj = joblib.load(path)
    if isinstance(obj, dict) and len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
        return next(iter(obj.values()))
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected dict in {path}, got {type(obj)!r}")


def _first(path_glob_root: Path, pattern: str) -> Path:
    matches = sorted(path_glob_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match {pattern} under {path_glob_root}")
    return matches[0]


def _quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)
    return np.stack(
        [
            np.stack([m00, m01, m02], axis=-1),
            np.stack([m10, m11, m12], axis=-1),
            np.stack([m20, m21, m22], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def _rot6(q: np.ndarray) -> np.ndarray:
    mat = _quat_to_matrix_xyzw(q)
    # Match SONIC/IsaacLab convention: first two matrix columns flattened.
    return mat[..., :, :2].reshape(*mat.shape[:-2], 6).astype(np.float32)


def _window(arr: np.ndarray, start: int, count: int, step: int) -> np.ndarray:
    idx = start + np.arange(count, dtype=np.int64) * step
    idx = np.clip(idx, 0, arr.shape[0] - 1)
    return arr[idx]


def _vel(x: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(np.asarray(x, dtype=np.float32), axis=0).astype(np.float32) * float(fps)


def _pad_repeat(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((dim,), dtype=np.float32)
    reps = math.ceil(dim / x.size)
    return np.tile(x, reps)[:dim].astype(np.float32)


def _encoder_obs_order(cfg: dict[str, Any]) -> list[str]:
    return [o["name"] for o in cfg["encoder"]["encoder_observations"] if o.get("enabled", True)]


def _policy_obs_order(cfg: dict[str, Any]) -> list[str]:
    return [o["name"] for o in cfg["observations"] if o.get("enabled", True)]


def build_encoder_input(order: list[str], robot: dict[str, Any], smpl: dict[str, Any], mode: int, start: int) -> np.ndarray:
    fps = float(robot.get("fps", 30))
    dof = np.asarray(robot["dof"], dtype=np.float32)
    dof_vel = _vel(dof, fps)
    root = np.asarray(robot["root_trans_offset"], dtype=np.float32)
    root_rot = np.asarray(robot["root_rot"], dtype=np.float32)
    smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
    smpl_root = np.asarray(smpl.get("transl", np.zeros((len(smpl_joints), 3), dtype=np.float32)), dtype=np.float32)
    smpl_pose = np.asarray(smpl.get("pose_aa", np.zeros((len(smpl_joints), 72), dtype=np.float32)), dtype=np.float32)
    smpl_quat_proxy = np.zeros((len(smpl_joints), 4), dtype=np.float32)
    smpl_quat_proxy[:, 3] = 1.0
    # The public SMPL sample lacks explicit root quaternions; use a conservative
    # identity proxy for that specific orientation-only field. The high-dimensional
    # SMPL joint trajectory remains real.

    vals: dict[str, np.ndarray] = {}
    vals["encoder_mode_4"] = np.asarray([mode, 0, 0, 0], dtype=np.float32)
    vals["motion_joint_positions_10frame_step5"] = _window(dof, start, 10, 5).reshape(-1)
    vals["motion_joint_velocities_10frame_step5"] = _window(dof_vel, start, 10, 5).reshape(-1)
    vals["motion_root_z_position_10frame_step5"] = _window(root[:, 2:3], start, 10, 5).reshape(-1)
    vals["motion_root_z_position"] = root[start : start + 1, 2].reshape(-1)
    vals["motion_anchor_orientation"] = _rot6(root_rot[start : start + 1]).reshape(-1)
    vals["motion_anchor_orientation_10frame_step5"] = _rot6(_window(root_rot, start, 10, 5)).reshape(-1)
    vals["motion_joint_positions_lowerbody_10frame_step5"] = _window(dof[:, LOWER_BODY_IDX], start, 10, 5).reshape(-1)
    vals["motion_joint_velocities_lowerbody_10frame_step5"] = _window(dof_vel[:, LOWER_BODY_IDX], start, 10, 5).reshape(-1)

    # Approximate VR 3-point fields from real SMPL joints (wrists + head), root-local.
    sj0 = smpl_joints[min(start, smpl_joints.shape[0] - 1)]
    root0 = smpl_root[min(start, smpl_root.shape[0] - 1)]
    point_idx = [20, 21, 15]  # left wrist, right wrist, head-ish in SMPL ordering.
    pts = sj0[point_idx] - root0[None, :]
    vals["vr_3point_local_target"] = pts.reshape(-1)
    vals["vr_3point_local_orn_target"] = np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), 3)
    vals["smpl_joints_10frame_step1"] = _window(smpl_joints, start, 10, 1).reshape(-1)
    vals["smpl_anchor_orientation_10frame_step1"] = _rot6(_window(smpl_quat_proxy, start, 10, 1)).reshape(-1)
    vals["motion_joint_positions_wrists_10frame_step1"] = _window(dof[:, WRIST_IDX], start, 10, 1).reshape(-1)

    chunks = []
    for name in order:
        if name not in vals:
            raise KeyError(f"No builder for encoder observation {name}")
        expected = ENCODER_DIMS[name]
        x = np.asarray(vals[name], dtype=np.float32).reshape(-1)
        if x.size != expected:
            raise ValueError(f"{name}: expected {expected}, got {x.size}")
        chunks.append(x)
    out = np.concatenate(chunks).astype(np.float32).reshape(1, -1)
    return out


def _quat_ang_vel(q: np.ndarray, fps: float) -> np.ndarray:
    # Conservative proxy: gradient of the vector part; enough for finite history ABI.
    return _vel(np.asarray(q, dtype=np.float32)[:, :3], fps)


def build_decoder_input(order: list[str], token: np.ndarray, robot: dict[str, Any], start: int) -> np.ndarray:
    fps = float(robot.get("fps", 30))
    dof = np.asarray(robot["dof"], dtype=np.float32)
    dof_vel = _vel(dof, fps)
    root_rot = np.asarray(robot["root_rot"], dtype=np.float32)
    base_ang = _quat_ang_vel(root_rot, fps)
    gravity = np.tile(np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32), (dof.shape[0], 1))
    last_actions = np.zeros_like(dof, dtype=np.float32)
    last_actions[1:] = dof[1:] - dof[:-1]
    vals = {
        "token_state": np.asarray(token, dtype=np.float32).reshape(64),
        "his_base_angular_velocity_10frame_step1": _window(base_ang, start, 10, 1).reshape(-1),
        "his_body_joint_positions_10frame_step1": _window(dof, start, 10, 1).reshape(-1),
        "his_body_joint_velocities_10frame_step1": _window(dof_vel, start, 10, 1).reshape(-1),
        "his_last_actions_10frame_step1": _window(last_actions, start, 10, 1).reshape(-1),
        "his_gravity_dir_10frame_step1": _window(gravity, start, 10, 1).reshape(-1),
    }
    chunks = []
    for name in order:
        expected = POLICY_DIMS[name]
        x = np.asarray(vals[name], dtype=np.float32).reshape(-1)
        if x.size != expected:
            raise ValueError(f"{name}: expected {expected}, got {x.size}")
        chunks.append(x)
    return np.concatenate(chunks).astype(np.float32).reshape(1, -1)


def stats(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float32)
    return {
        "shape": list(x.shape),
        "finite": bool(np.isfinite(x).all()),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "abs_max": float(np.nanmax(np.abs(x))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--starts", type=int, nargs="*", default=[0, 25, 50, 75, 100, 150])
    args = ap.parse_args()

    assets = args.assets
    cfg = yaml.safe_load((assets / "observation_config.yaml").read_text())
    enc_order = _encoder_obs_order(cfg)
    pol_order = _policy_obs_order(cfg)
    expected_enc_dim = sum(ENCODER_DIMS[n] for n in enc_order)
    expected_dec_dim = sum(POLICY_DIMS[n] for n in pol_order)

    robot = _nested_motion(_first(assets / "sample_data", "robot_filtered/210531/*.pkl"))
    smpl = _nested_motion(_first(assets / "sample_data", "smpl_filtered/*.pkl"))

    enc_sess = ort.InferenceSession(str(assets / "model_encoder.onnx"), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(assets / "model_decoder.onnx"), providers=["CPUExecutionProvider"])
    enc_in = enc_sess.get_inputs()[0]
    enc_out = enc_sess.get_outputs()[0]
    dec_in = dec_sess.get_inputs()[0]
    dec_out = dec_sess.get_outputs()[0]

    checks: list[str] = []
    failures: list[str] = []
    if enc_in.name != "obs_dict" or dec_in.name != "obs_dict":
        failures.append("ONNX input tensor name is not obs_dict")
    if list(enc_in.shape) != [1, expected_enc_dim]:
        failures.append(f"encoder shape {enc_in.shape} != [1,{expected_enc_dim}]")
    if list(enc_out.shape) != [1, 64]:
        failures.append(f"encoder output shape {enc_out.shape} != [1,64]")
    if list(dec_in.shape) != [1, expected_dec_dim]:
        failures.append(f"decoder shape {dec_in.shape} != [1,{expected_dec_dim}]")
    if list(dec_out.shape) != [1, 29]:
        failures.append(f"decoder output shape {dec_out.shape} != [1,29]")

    token_rows = []
    action_rows = []
    by_mode: dict[str, dict[str, Any]] = {}
    modes = {"g1": 0, "teleop": 1, "smpl": 2}
    max_start = min(np.asarray(robot["dof"]).shape[0], np.asarray(smpl["smpl_joints"]).shape[0]) - 60
    starts = [s for s in args.starts if 0 <= s <= max_start]
    if len(starts) < 3:
        failures.append(f"too few valid starts for semantic variation: {starts}")

    for mode_name, mode_id in modes.items():
        mode_tokens = []
        mode_actions = []
        for start in starts:
            x = build_encoder_input(enc_order, robot, smpl, mode_id, start)
            if x.shape != (1, expected_enc_dim) or not np.isfinite(x).all():
                failures.append(f"bad encoder input for {mode_name}@{start}: {x.shape}, finite={np.isfinite(x).all()}")
                continue
            token = enc_sess.run([enc_out.name], {enc_in.name: x})[0].astype(np.float32)
            dec_x = build_decoder_input(pol_order, token.reshape(-1), robot, start)
            action = dec_sess.run([dec_out.name], {dec_in.name: dec_x})[0].astype(np.float32)
            mode_tokens.append(token.reshape(-1))
            mode_actions.append(action.reshape(-1))
        mt = np.stack(mode_tokens, axis=0) if mode_tokens else np.empty((0, 64), dtype=np.float32)
        ma = np.stack(mode_actions, axis=0) if mode_actions else np.empty((0, 29), dtype=np.float32)
        by_mode[mode_name] = {"tokens": stats(mt) if mt.size else {}, "actions": stats(ma) if ma.size else {}}
        token_rows.append(mt)
        action_rows.append(ma)

        if not mt.size or not np.isfinite(mt).all():
            failures.append(f"{mode_name}: non-finite or missing tokens")
        elif float(np.std(mt)) < 1e-5:
            failures.append(f"{mode_name}: token std too small ({np.std(mt):.3g})")
        elif float(np.mean(np.std(mt, axis=0))) < 1e-6:
            failures.append(f"{mode_name}: tokens do not vary across real frames")
        else:
            checks.append(f"{mode_name}: finite varying tokens")

        if not ma.size or not np.isfinite(ma).all():
            failures.append(f"{mode_name}: non-finite or missing decoder actions")
        elif float(np.max(np.abs(ma))) > 100.0:
            failures.append(f"{mode_name}: decoder action magnitude pathological ({np.max(np.abs(ma)):.3g})")
        elif float(np.std(ma)) < 1e-7:
            failures.append(f"{mode_name}: decoder actions are effectively constant")
        else:
            checks.append(f"{mode_name}: finite bounded decoder actions")

    all_tokens = np.concatenate(token_rows, axis=0) if token_rows else np.empty((0, 64), dtype=np.float32)
    all_actions = np.concatenate(action_rows, axis=0) if action_rows else np.empty((0, 29), dtype=np.float32)
    if all_tokens.size and len(token_rows) == 3:
        means = [x.mean(axis=0) for x in token_rows]
        pair_d = [float(np.linalg.norm(means[i] - means[j])) for i in range(3) for j in range(i + 1, 3)]
        if max(pair_d) < 1e-4:
            failures.append(f"mode means are indistinguishable: {pair_d}")
        else:
            checks.append("encoder mode affects tokens")
    else:
        pair_d = []

    result = {
        "gate": "sonic_semantic_token_assets",
        "assets": str(assets),
        "onnxruntime_providers": ort.get_available_providers(),
        "encoder_order": enc_order,
        "policy_order": pol_order,
        "expected_encoder_dim": expected_enc_dim,
        "expected_decoder_dim": expected_dec_dim,
        "starts": starts,
        "by_mode": by_mode,
        "all_tokens": stats(all_tokens) if all_tokens.size else {},
        "all_decoder_actions": stats(all_actions) if all_actions.size else {},
        "mode_mean_pairwise_l2": pair_d,
        "checks": checks,
        "failures": failures,
        "pass": not failures,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
