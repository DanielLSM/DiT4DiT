#!/usr/bin/env python3
"""Offline metrics + deployment stats for the DiT4DiT SONIC real-token pilot.

This gate answers a narrower question than full robot evaluation: given the current
real-token LeRobot dataset and a trained FlowmatchingActionHead checkpoint, can we
export the action normalization contract, unnormalize predictions consistently, and
produce per-slice metrics/videos for the SONIC 64+7+7 action ABI?

It intentionally supports the lightweight pilot checkpoint produced by
`scripts/train_sonic_real_token_action_head.py` where the VLM embeddings are
synthetic. That means the metrics are a structural regression gate, not a claim of
visual-language task competence. Boring distinction; important distinction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

ACTION_DIM = 78
MOTION_TOKEN_DIM = 64
HAND_DIM = 7
MOTION_SLICE = slice(0, 64)
LEFT_HAND_SLICE = slice(64, 71)
RIGHT_HAND_SLICE = slice(71, 78)


def _as_array(values: Any, expected: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape != (expected,):
        raise ValueError(f"{name}: expected {expected} values, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contains non-finite values")
    return arr


def split_sonic_action(action: np.ndarray) -> dict[str, np.ndarray]:
    """Split an arbitrary `[..., 78]` SONIC action tensor into canonical fields."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected last dim {ACTION_DIM}, got shape {action.shape}")
    return {
        "motion_token": action[..., MOTION_SLICE],
        "left_hand_joints": action[..., LEFT_HAND_SLICE],
        "right_hand_joints": action[..., RIGHT_HAND_SLICE],
    }


def _slice_stats(action_stats: dict[str, Any], sl: slice, mode: str, name: str) -> dict[str, Any]:
    start = int(sl.start or 0)
    end = int(sl.stop or ACTION_DIM)
    width = end - start
    out = {
        "mode": mode,
        "start": start,
        "end": end,
        "dim": width,
        "min": _as_array(action_stats["min"], ACTION_DIM, "action.min")[sl].astype(float).tolist(),
        "max": _as_array(action_stats["max"], ACTION_DIM, "action.max")[sl].astype(float).tolist(),
        "q01": _as_array(action_stats.get("q01", action_stats["min"]), ACTION_DIM, "action.q01")[sl].astype(float).tolist(),
        "q99": _as_array(action_stats.get("q99", action_stats["max"]), ACTION_DIM, "action.q99")[sl].astype(float).tolist(),
    }
    if "mean" in action_stats:
        out["mean"] = _as_array(action_stats["mean"], ACTION_DIM, "action.mean")[sl].astype(float).tolist()
    if "std" in action_stats:
        out["std"] = _as_array(action_stats["std"], ACTION_DIM, "action.std")[sl].astype(float).tolist()
    # Deployment code can use this to clamp placeholder hands without extra policy.
    mn = np.asarray(out["min"], dtype=np.float32)
    mx = np.asarray(out["max"], dtype=np.float32)
    out["constant_dims"] = (mn == mx).astype(bool).tolist()
    out["field"] = name
    return out


def export_sonic_action_stats(dataset_dir: Path, output_json: Path, motion_mode: str = "min_max") -> dict[str, Any]:
    """Export a plain-JSON per-key normalization contract from LeRobot stats.json."""
    stats_path = dataset_dir / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"missing dataset stats: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    action_stats = stats.get("action")
    if not isinstance(action_stats, dict):
        raise ValueError(f"{stats_path} has no action stats block")
    for key in ("min", "max"):
        _as_array(action_stats[key], ACTION_DIM, f"action.{key}")
    result = {
        "format": "sonic_action_stats_v1",
        "dataset_dir": str(dataset_dir),
        "source_stats_json": str(stats_path),
        "action_dim": ACTION_DIM,
        "fields": ["action.motion_token", "action.left_hand_joints", "action.right_hand_joints"],
        "action.motion_token": _slice_stats(action_stats, MOTION_SLICE, motion_mode, "action.motion_token"),
        "action.left_hand_joints": _slice_stats(action_stats, LEFT_HAND_SLICE, "min_max", "action.left_hand_joints"),
        "action.right_hand_joints": _slice_stats(action_stats, RIGHT_HAND_SLICE, "min_max", "action.right_hand_joints"),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _inverse_field(x: np.ndarray, field_stats: dict[str, Any]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mode = field_stats["mode"]
    if mode == "min_max":
        lo = np.asarray(field_stats["min"], dtype=np.float32)
        hi = np.asarray(field_stats["max"], dtype=np.float32)
        return (x + 1.0) * 0.5 * (hi - lo) + lo
    if mode == "q99":
        lo = np.asarray(field_stats["q01"], dtype=np.float32)
        hi = np.asarray(field_stats["q99"], dtype=np.float32)
        return (x + 1.0) * 0.5 * (hi - lo) + lo
    if mode == "mean_std":
        mean = np.asarray(field_stats["mean"], dtype=np.float32)
        std = np.asarray(field_stats["std"], dtype=np.float32)
        return x * std + mean
    raise ValueError(f"unsupported normalization mode: {mode}")


def unnormalize_sonic_action(normalized_action: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Map normalized `[..., 78]` SONIC actions back to raw per-field values."""
    normalized_action = np.asarray(normalized_action, dtype=np.float32)
    if normalized_action.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected last dim {ACTION_DIM}, got {normalized_action.shape}")
    raw = np.empty_like(normalized_action, dtype=np.float32)
    raw[..., MOTION_SLICE] = _inverse_field(normalized_action[..., MOTION_SLICE], stats["action.motion_token"])
    raw[..., LEFT_HAND_SLICE] = _inverse_field(normalized_action[..., LEFT_HAND_SLICE], stats["action.left_hand_joints"])
    raw[..., RIGHT_HAND_SLICE] = _inverse_field(normalized_action[..., RIGHT_HAND_SLICE], stats["action.right_hand_joints"])
    return raw


def compute_action_metrics(pred: np.ndarray, target: np.ndarray, motion_token_abs_limit: float | None = None) -> dict[str, float]:
    """Compute per-slice MSE/range/smoothness metrics for `[T, 78]` or `[N, T, 78]`."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    if pred.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected action dim {ACTION_DIM}, got {pred.shape}")
    err = pred - target
    parts = split_sonic_action(err)
    token = split_sonic_action(pred)["motion_token"]
    if motion_token_abs_limit is None:
        motion_token_abs_limit = float(np.nanmax(np.abs(split_sonic_action(target)["motion_token"])))
    if not math.isfinite(motion_token_abs_limit) or motion_token_abs_limit <= 0:
        motion_token_abs_limit = 1.0
    diffs = np.diff(pred, axis=-2) if pred.shape[-2] > 1 else np.zeros_like(pred[..., :0, :])
    smooth = np.linalg.norm(diffs, axis=-1) if diffs.size else np.zeros((1,), dtype=np.float32)
    return {
        "mse_total": float(np.mean(err**2)),
        "mse_motion_token": float(np.mean(parts["motion_token"] ** 2)),
        "mse_left_hand_joints": float(np.mean(parts["left_hand_joints"] ** 2)),
        "mse_right_hand_joints": float(np.mean(parts["right_hand_joints"] ** 2)),
        "mae_total": float(np.mean(np.abs(err))),
        "motion_token_abs_max": float(np.max(np.abs(token))),
        "motion_token_abs_limit": float(motion_token_abs_limit),
        "motion_token_range_violation_rate": float(np.mean(np.abs(token) > motion_token_abs_limit)),
        "temporal_smoothness_l2_mean": float(np.mean(smooth)),
        "temporal_smoothness_l2_max": float(np.max(smooth)),
    }


def render_eval_video(pred: np.ndarray, target: np.ndarray, output_video: Path, fps: int = 10) -> dict[str, Any]:
    """Render a compact predicted-vs-target trace video for Telegram/GitHub inspection."""
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "reason": f"cv2 unavailable: {exc}"}
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    h, w = 520, 760
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    use_imageio_fallback = not writer.isOpened()
    frames_for_imageio: list[np.ndarray] = []
    if use_imageio_fallback:
        writer.release()

    pred_parts = split_sonic_action(pred)
    target_parts = split_sonic_action(target)
    series = {
        "motion token L2": (np.linalg.norm(pred_parts["motion_token"], axis=-1), np.linalg.norm(target_parts["motion_token"], axis=-1)),
        "left hand mean": (pred_parts["left_hand_joints"].mean(axis=-1), target_parts["left_hand_joints"].mean(axis=-1)),
        "right hand mean": (pred_parts["right_hand_joints"].mean(axis=-1), target_parts["right_hand_joints"].mean(axis=-1)),
    }

    def draw_pair(img: np.ndarray, y0: int, label: str, pred_s: np.ndarray, tgt_s: np.ndarray) -> None:
        cv2.putText(img, label, (34, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        cv2.rectangle(img, (34, y0), (726, y0 + 105), (210, 210, 210), 1)
        vals = np.concatenate([pred_s.reshape(-1), tgt_s.reshape(-1)])
        lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
        if math.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        xs = np.linspace(48, 712, num=len(pred_s)).astype(np.int32)
        for arr, color in ((tgt_s, (80, 150, 80)), (pred_s, (40, 80, 220))):
            ys = (y0 + 95 - np.clip((arr - lo) / (hi - lo), 0, 1) * 82).astype(np.int32)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)

    for t in range(pred.shape[0]):
        img = np.full((h, w, 3), 245, dtype=np.uint8)
        cv2.putText(img, "SONIC real-token checkpoint offline eval", (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 25, 25), 2)
        cv2.putText(img, "blue=prediction green=target", (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (65, 65, 65), 1)
        for idx, (label, (pred_s, tgt_s)) in enumerate(series.items()):
            draw_pair(img, 115 + idx * 125, label, pred_s, tgt_s)
        x = int(48 + (712 - 48) * t / max(1, pred.shape[0] - 1))
        cv2.line(img, (x, 100), (x, 485), (0, 0, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"frame {t:02d}/{pred.shape[0]-1}", (610, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (65, 65, 65), 1)
        if use_imageio_fallback:
            frames_for_imageio.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            writer.write(img)
    if use_imageio_fallback:
        try:
            import imageio.v2 as imageio  # type: ignore

            imageio.mimsave(str(output_video), frames_for_imageio, fps=fps, macro_block_size=1)
            backend = "imageio_ffmpeg"
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "reason": f"cv2 writer unavailable and imageio fallback failed: {exc}"}
    else:
        writer.release()
        backend = "cv2_mp4v"
    return {"ok": True, "path": str(output_video), "frames": int(pred.shape[0]), "fps": fps, "writer": backend}


def _load_checkpoint_model(checkpoint: Path, horizon: int, device: Any):
    import torch

    from scripts.train_sonic_real_token_action_head import _build_action_head

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be dict, got {type(ckpt)!r}")
    raw_args = ckpt.get("args")
    args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
    hidden_size = int(args.get("hidden_size", 256))
    layers = int(args.get("layers", 1))
    for key, expected in (("horizon", horizon), ("action_dim", ACTION_DIM), ("state_dim", 29)):
        actual = ckpt.get(key)
        if actual is not None and int(actual) != int(expected):
            raise AssertionError(f"checkpoint {key}={actual} does not match expected {expected}")
    model = _build_action_head(ACTION_DIM, 29, horizon, hidden_size, layers, device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        raise AssertionError(f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, {"path": str(checkpoint), "hidden_size": hidden_size, "layers": layers, "checkpoint_steps": args.get("steps"), "dataset_len": ckpt.get("dataset_len")}


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset

    dataset_dir = args.dataset_dir or (args.data_root / args.dataset_name)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = export_sonic_action_stats(dataset_dir, output_dir / "sonic_action_stats.json", motion_mode=args.motion_mode)

    data_cfg = OmegaConf.create(
        {
            "data_root_dir": str(args.data_root),
            "data_mix": args.data_mix,
            "delete_pause_frame": False,
            "lerobot_version": "v2.0",
            "video_backend": args.video_backend,
            "video_delta_indices": [0, 1],
            "max_state_dim": 29,
            "max_action_dim": ACTION_DIM,
            "action_video_freq_ratio": 1,
        }
    )
    dataset = get_vla_dataset(data_cfg=data_cfg, mode="train", seed=args.seed)
    if len(dataset) <= 0:
        raise AssertionError("dataset has no samples")
    sample_count = min(args.sample_count, len(dataset))
    start = min(max(args.start_index, 0), len(dataset) - sample_count)
    samples = [dataset[i] for i in range(start, start + sample_count)]
    actions_norm = np.stack([s["action"] for s in samples]).astype(np.float32)
    states = np.stack([s["state"] for s in samples]).astype(np.float32)
    if actions_norm.shape[-1] != ACTION_DIM or states.shape[-1] != 29:
        raise AssertionError(f"bad dataset shapes: actions={actions_norm.shape}, states={states.shape}")
    horizon = int(actions_norm.shape[1])
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, checkpoint_meta = _load_checkpoint_model(args.checkpoint, horizon, device)

    t0 = time.time()
    preds_norm = []
    with torch.no_grad():
        for i in range(sample_count):
            torch.manual_seed(args.seed + i)
            vl_embs = torch.randn((1, args.vl_tokens, 768), device=device, dtype=torch.float32)
            state_t = torch.as_tensor(states[i : i + 1], device=device, dtype=torch.float32)
            pred = model.predict_action(vl_embs, state_t)[0].detach().cpu().numpy().astype(np.float32)
            preds_norm.append(pred)
    preds_norm_arr = np.stack(preds_norm, axis=0)
    targets_raw = unnormalize_sonic_action(actions_norm, stats)
    preds_raw = unnormalize_sonic_action(preds_norm_arr, stats)
    limit = max(float(np.max(np.abs(targets_raw[..., MOTION_SLICE]))), 1e-6)
    metrics_norm = compute_action_metrics(preds_norm_arr, actions_norm, motion_token_abs_limit=1.0)
    metrics_raw = compute_action_metrics(preds_raw, targets_raw, motion_token_abs_limit=limit)

    npz_path = output_dir / "offline_eval_predictions.npz"
    np.savez_compressed(
        npz_path,
        pred_normalized=preds_norm_arr,
        target_normalized=actions_norm,
        pred_raw=preds_raw,
        target_raw=targets_raw,
        states=states,
        sample_indices=np.arange(start, start + sample_count, dtype=np.int64),
    )
    video = None
    if args.output_video:
        video_path = args.output_video if args.output_video.is_absolute() else output_dir / args.output_video
        video = render_eval_video(preds_raw[0], targets_raw[0], video_path, fps=args.video_fps)

    result = {
        "ok": True,
        "gate": "sonic_real_token_checkpoint_offline_eval",
        "data_root": str(args.data_root),
        "dataset_dir": str(dataset_dir),
        "data_mix": args.data_mix,
        "checkpoint": checkpoint_meta,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_len": len(dataset),
        "sample_start": start,
        "sample_count": sample_count,
        "horizon": horizon,
        "action_dim": ACTION_DIM,
        "stats_json": str(output_dir / "sonic_action_stats.json"),
        "predictions_npz": str(npz_path),
        "metrics_normalized": metrics_norm,
        "metrics_raw": metrics_raw,
        "video": video,
        "elapsed_sec": time.time() - t0,
        "interpretation": "pilot action-head structural eval; VLM embeddings are synthetic, so this is not visual-language task competence",
    }
    summary_path = output_dir / "offline_eval_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-name", default="sonic_g1_real_token_lerobot")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--data-mix", default="sonic_g1_real_token")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-count", type=int, default=8)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--motion-mode", choices=["min_max", "q99", "mean_std"], default="min_max")
    p.add_argument("--vl-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--video-backend", default="torchvision_av", choices=["decord", "opencv", "torchvision_av", "torchcodec"])
    p.add_argument("--output-video", type=Path, default=Path("offline_eval_trace.mp4"))
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
