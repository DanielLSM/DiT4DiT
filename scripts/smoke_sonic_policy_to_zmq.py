#!/usr/bin/env python3
"""Gate: DiT4DiT 78D policy prediction -> SONIC latent protocol-v4 packets.

This is deliberately one gate past the action-head smoke: it loads the SONIC G1
LeRobot fixture, trains the lightweight DiT action head for a few optimizer
steps, predicts a 40-step 78D action chunk, splits it according to the SONIC ABI
(64 motion-token dims + 7 left hand + 7 right hand), packs every step into the
same single-frame latent v4 wire format consumed by SONIC's C++ ZMQ subscriber,
and round-trips the bytes back to numpy.

It does not run the SONIC decoder/physics controller. It proves the VLA output is
structurally acceptable to the SONIC action ingress.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset
from DiT4DiT.model.modules.action_model.ActionDiT import FlowmatchingActionHead


HEADER_SIZE = 1280
ACTION_DIM = 78
MOTION_TOKEN_DIM = 64
HAND_DIM = 7


def _ns(mapping: dict[str, Any]) -> SimpleNamespace:
    out = SimpleNamespace()
    for key, value in mapping.items():
        setattr(out, key, _ns(value) if isinstance(value, dict) else value)
    return out


def _build_action_head(
    action_dim: int,
    state_dim: int,
    horizon: int,
    device: torch.device,
    hidden_size: int = 256,
    layers: int = 1,
    inference_steps: int = 2,
) -> FlowmatchingActionHead:
    cfg = _ns(
        {
            "framework": {
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": hidden_size,
                    "add_pos_embed": True,
                    "max_seq_len": 128,
                    "action_dim": action_dim,
                    "state_dim": state_dim,
                    "future_action_window_size": horizon - 1,
                    "action_horizon": horizon,
                    "past_action_window_size": 0,
                    "repeated_diffusion_steps": 1,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "num_inference_timesteps": inference_steps,
                    "num_target_vision_tokens": 32,
                }
            }
        }
    )
    cfg.framework.action_model.diffusion_model_cfg = {
        "cross_attention_dim": 768,
        "dropout": 0.0,
        "final_dropout": False,
        "interleave_self_attention": False,
        "norm_type": "ada_norm",
        "num_layers": layers,
        "output_dim": hidden_size,
        "positional_embeddings": None,
    }
    return FlowmatchingActionHead(cfg).to(device)


def _build_header(fields: list[dict[str, Any]], version: int = 4, count: int = 1) -> bytes:
    header = {"v": version, "endian": "le", "count": count, "fields": fields}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > HEADER_SIZE:
        raise ValueError(f"SONIC header too large: {len(encoded)} > {HEADER_SIZE}")
    return encoded.ljust(HEADER_SIZE, b"\x00")


def pack_sonic_latent_v4(action_row: np.ndarray, frame_index: int) -> bytes:
    """Pack one 78D action into SONIC's latent protocol v4 single-message frame."""
    action_row = np.asarray(action_row, dtype=np.float32)
    if action_row.shape != (ACTION_DIM,):
        raise ValueError(f"expected one 78D action row, got {action_row.shape}")
    if not np.isfinite(action_row).all():
        raise ValueError("action row contains non-finite values")

    token_state = np.ascontiguousarray(action_row[:MOTION_TOKEN_DIM].reshape(1, MOTION_TOKEN_DIM), dtype=np.float32)
    left_hand = np.ascontiguousarray(action_row[MOTION_TOKEN_DIM : MOTION_TOKEN_DIM + HAND_DIM].reshape(1, HAND_DIM), dtype=np.float32)
    right_hand = np.ascontiguousarray(action_row[MOTION_TOKEN_DIM + HAND_DIM :].reshape(1, HAND_DIM), dtype=np.float32)
    frame = np.asarray([frame_index], dtype=np.int64)

    fields = [
        {"name": "token_state", "dtype": "f32", "shape": list(token_state.shape)},
        {"name": "frame_index", "dtype": "i64", "shape": list(frame.shape)},
        {"name": "left_hand_joints", "dtype": "f32", "shape": list(left_hand.shape)},
        {"name": "right_hand_joints", "dtype": "f32", "shape": list(right_hand.shape)},
    ]
    payload = b"".join([token_state.tobytes(), frame.tobytes(), left_hand.tobytes(), right_hand.tobytes()])
    return b"pose" + _build_header(fields, version=4, count=1) + payload


def unpack_sonic_latent_message(message: bytes) -> dict[str, np.ndarray]:
    if not message.startswith(b"pose"):
        raise ValueError("message does not start with SONIC 'pose' topic")
    raw = message[len(b"pose") :]
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"short message: {len(raw)} < {HEADER_SIZE}")
    header_len = raw[:HEADER_SIZE].find(b"\x00")
    if header_len < 0:
        header_len = HEADER_SIZE
    header = json.loads(raw[:header_len].decode("utf-8"))
    if header.get("v") != 4:
        raise ValueError(f"expected protocol v4, got {header.get('v')}")
    if header.get("endian") != "le":
        raise ValueError(f"expected little-endian payload, got {header.get('endian')}")

    dtype_map = {"f32": np.dtype("<f4"), "i64": np.dtype("<i8"), "i32": np.dtype("<i4"), "u8": np.dtype("u1")}
    payload = raw[HEADER_SIZE:]
    out: dict[str, np.ndarray] = {}
    offset = 0
    for field in header["fields"]:
        dtype = dtype_map[field["dtype"]]
        shape = tuple(int(x) for x in field["shape"])
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if offset + nbytes > len(payload):
            raise ValueError(f"field {field['name']} overruns payload")
        out[field["name"]] = np.frombuffer(payload[offset : offset + nbytes], dtype=dtype).reshape(shape).copy()
        offset += nbytes
    if offset != len(payload):
        raise ValueError(f"payload has trailing bytes: {len(payload) - offset}")
    return out


def publish_sonic_latent_messages(
    messages: list[bytes],
    bind_host: str,
    port: int,
    rate_hz: float = 50.0,
    warmup_sec: float = 1.0,
) -> dict[str, Any]:
    """Publish pre-packed SONIC latent-v4 messages over a real ZMQ PUB socket."""
    try:
        import zmq  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime image
        return {"ok": False, "reason": f"pyzmq unavailable: {exc}"}

    endpoint = f"tcp://{bind_host}:{port}"
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    try:
        socket.setsockopt(zmq.SNDHWM, max(10, len(messages)))
        socket.bind(endpoint)
        time.sleep(max(0.0, warmup_sec))  # PUB/SUB slow-joiner guard. Ugly, necessary.
        interval = 0.0 if rate_hz <= 0 else 1.0 / rate_hz
        send_times = []
        for msg in messages:
            socket.send(msg)
            send_times.append(time.time())
            if interval > 0:
                time.sleep(interval)
    finally:
        socket.close(linger=0)
        context.term()

    return {
        "ok": True,
        "endpoint": endpoint,
        "message_count": len(messages),
        "rate_hz": rate_hz,
        "warmup_sec": warmup_sec,
        "first_send_time": send_times[0] if send_times else None,
        "last_send_time": send_times[-1] if send_times else None,
    }


def render_prediction_video(action: np.ndarray, output_video: Path, fps: int = 10) -> dict[str, Any]:
    """Create a small diagnostic MP4 of policy token/hand traces if OpenCV exists."""
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime image
        return {"ok": False, "reason": f"cv2 unavailable: {exc}"}

    action = np.asarray(action, dtype=np.float32)
    token = action[:, :MOTION_TOKEN_DIM]
    left = action[:, MOTION_TOKEN_DIM : MOTION_TOKEN_DIM + HAND_DIM]
    right = action[:, MOTION_TOKEN_DIM + HAND_DIM :]
    h, w = 480, 720
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frames_for_imageio: list[np.ndarray] = []
    use_imageio_fallback = not writer.isOpened()
    if use_imageio_fallback:
        writer.release()

    def draw_series(img: np.ndarray, series: np.ndarray, x0: int, y0: int, width: int, height: int, color: tuple[int, int, int]) -> None:
        vals = np.asarray(series, dtype=np.float32)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return
        lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
        if math.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        xs = np.linspace(x0, x0 + width - 1, num=len(vals)).astype(np.int32)
        ys = (y0 + height - 1 - np.clip((vals - lo) / (hi - lo), 0, 1) * (height - 1)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)

    energy = np.linalg.norm(token, axis=1)
    for t in range(action.shape[0]):
        img = np.full((h, w, 3), 245, dtype=np.uint8)
        cv2.putText(img, "DiT4DiT -> SONIC latent v4 prediction smoke", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2)
        cv2.putText(img, f"frame {t:02d}/{action.shape[0]-1} | 64 token dims + 7L + 7R", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
        cv2.rectangle(img, (40, 105), (680, 245), (220, 220, 220), 1)
        cv2.putText(img, "motion-token L2 energy", (48, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)
        draw_series(img, energy, 55, 145, 610, 85, (30, 90, 220))
        x = int(55 + (610 - 1) * t / max(1, action.shape[0] - 1))
        cv2.line(img, (x, 140), (x, 235), (0, 0, 180), 1, cv2.LINE_AA)

        cv2.rectangle(img, (40, 275), (680, 440), (220, 220, 220), 1)
        cv2.putText(img, "hand joint channels: left blue, right orange", (48, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)
        for j in range(HAND_DIM):
            draw_series(img, left[:, j], 55, 315, 610, 50, (220, 80, 30))
            draw_series(img, right[:, j], 55, 375, 610, 50, (40, 145, 230))
        cv2.line(img, (x, 310), (x, 430), (0, 0, 180), 1, cv2.LINE_AA)
        if use_imageio_fallback:
            frames_for_imageio.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            writer.write(img)
    if use_imageio_fallback:
        try:
            import imageio.v2 as imageio  # type: ignore

            imageio.mimsave(str(output_video), frames_for_imageio, fps=fps, macro_block_size=1)
            writer_backend = "imageio_ffmpeg"
        except Exception as exc:  # pragma: no cover - depends on runtime image
            return {"ok": False, "reason": f"cv2 writer unavailable and imageio fallback failed: {exc}"}
    else:
        writer.release()
        writer_backend = "cv2_mp4v"
    return {"ok": True, "path": str(output_video), "frames": int(action.shape[0]), "fps": fps, "writer": writer_backend}


def run_gate(
    data_root: Path,
    data_mix: str,
    steps: int,
    batch_size: int,
    device: torch.device,
    video_backend: str,
    output_npz: Path | None,
    output_video: Path | None,
    checkpoint: Path | None = None,
    hidden_size: int = 256,
    layers: int = 1,
    inference_steps: int = 2,
    publish_zmq_host: str | None = None,
    publish_zmq_port: int | None = None,
    publish_rate_hz: float = 50.0,
    publish_warmup_sec: float = 1.0,
) -> dict[str, Any]:
    torch.manual_seed(123)
    np.random.seed(123)
    data_cfg = OmegaConf.create(
        {
            "data_root_dir": str(data_root),
            "data_mix": data_mix,
            "delete_pause_frame": False,
            "lerobot_version": "v2.0",
            "video_backend": video_backend,
            "video_delta_indices": [0, 1],
            "max_state_dim": 29,
            "max_action_dim": ACTION_DIM,
            "action_video_freq_ratio": 1,
        }
    )
    dataset = get_vla_dataset(data_cfg=data_cfg, mode="train", seed=7)
    if len(dataset) <= 0:
        raise AssertionError("dataset has no samples")

    samples = [dataset[i] for i in range(batch_size)]
    actions = np.stack([s["action"] for s in samples]).astype(np.float32)
    states = np.stack([s["state"] for s in samples]).astype(np.float32)
    masks = np.stack([s["action_mask"] for s in samples]).astype(np.float32)
    if actions.shape[-1] != ACTION_DIM:
        raise AssertionError(f"expected 78D action, got {actions.shape}")
    if states.shape[-1] != 29:
        raise AssertionError(f"expected 29D G1 state, got {states.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise AssertionError("actions/states contain non-finite values")

    horizon = int(actions.shape[1])
    checkpoint_meta: dict[str, Any] | None = None
    ckpt: dict[str, Any] | None = None
    if checkpoint is not None:
        loaded = torch.load(checkpoint, map_location=device, weights_only=False)
        if not isinstance(loaded, dict):
            raise AssertionError(f"checkpoint must be a dict, got {type(loaded)!r}")
        ckpt = loaded
        raw_args = ckpt.get("args")
        ckpt_args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        checkpoint_meta = {
            "path": str(checkpoint),
            "dataset_len": ckpt.get("dataset_len"),
            "horizon": ckpt.get("horizon"),
            "action_dim": ckpt.get("action_dim"),
            "state_dim": ckpt.get("state_dim"),
            "steps": ckpt_args.get("steps"),
            "hidden_size": ckpt_args.get("hidden_size"),
            "layers": ckpt_args.get("layers"),
        }
        hidden_size = int(checkpoint_meta["hidden_size"] or hidden_size)
        layers = int(checkpoint_meta["layers"] or layers)
        for key, expected in (("horizon", horizon), ("action_dim", ACTION_DIM), ("state_dim", int(states.shape[-1]))):
            actual = checkpoint_meta.get(key)
            if actual is not None and int(actual) != int(expected):
                raise AssertionError(f"checkpoint {key}={actual} does not match runtime {expected}")

    model = _build_action_head(
        action_dim=ACTION_DIM,
        state_dim=int(states.shape[-1]),
        horizon=horizon,
        device=device,
        hidden_size=hidden_size,
        layers=layers,
        inference_steps=inference_steps,
    )
    if ckpt is not None:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            raise AssertionError(f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    action_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
    state_t = torch.as_tensor(states, device=device, dtype=torch.float32)
    mask_t = torch.as_tensor(masks, device=device, dtype=torch.float32)
    vl_embs = torch.randn((batch_size, 8, 768), device=device, dtype=torch.float32)

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(vl_embs, action_t, mask_t, state_t)
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite loss: {loss}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        pred = model.predict_action(vl_embs[:1], state_t[:1])[0].detach().cpu().numpy().astype(np.float32)
    if pred.shape != (horizon, ACTION_DIM):
        raise AssertionError(f"bad prediction shape {pred.shape}")
    if not np.isfinite(pred).all():
        raise AssertionError("prediction contains non-finite values")

    messages = [pack_sonic_latent_v4(row, i) for i, row in enumerate(pred)]
    decoded = [unpack_sonic_latent_message(msg) for msg in messages]
    round_trip_errors = []
    for i, row in enumerate(pred):
        joined = np.concatenate(
            [decoded[i]["token_state"].reshape(-1), decoded[i]["left_hand_joints"].reshape(-1), decoded[i]["right_hand_joints"].reshape(-1)]
        ).astype(np.float32)
        round_trip_errors.append(float(np.max(np.abs(joined - row))))
        if int(decoded[i]["frame_index"][0]) != i:
            raise AssertionError(f"frame index mismatch at {i}: {decoded[i]['frame_index']}")
    message_lengths = np.asarray([len(m) for m in messages], dtype=np.int64)

    if output_npz is not None:
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            prediction=pred,
            target=actions[0],
            state=states[0],
            message_lengths=message_lengths,
            first_message=np.frombuffer(messages[0], dtype=np.uint8),
        )

    video_result = None
    if output_video is not None:
        video_result = render_prediction_video(pred, output_video)

    zmq_publish_result = None
    if publish_zmq_port is not None:
        zmq_publish_result = publish_sonic_latent_messages(
            messages,
            bind_host=publish_zmq_host or "*",
            port=publish_zmq_port,
            rate_hz=publish_rate_hz,
            warmup_sec=publish_warmup_sec,
        )

    token = pred[:, :MOTION_TOKEN_DIM]
    left = pred[:, MOTION_TOKEN_DIM : MOTION_TOKEN_DIM + HAND_DIM]
    right = pred[:, MOTION_TOKEN_DIM + HAND_DIM :]
    return {
        "ok": True,
        "gate": "dit4dit_policy_prediction_to_sonic_latent_v4_wire",
        "data_root": str(data_root),
        "data_mix": data_mix,
        "dataset_len": len(dataset),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_action_shape": list(actions.shape),
        "batch_state_shape": list(states.shape),
        "prediction_shape": list(pred.shape),
        "sonic_abi": {"action_dim": ACTION_DIM, "motion_token": MOTION_TOKEN_DIM, "left_hand": HAND_DIM, "right_hand": HAND_DIM},
        "checkpoint": checkpoint_meta,
        "action_head_config": {"hidden_size": hidden_size, "layers": layers, "inference_steps": inference_steps},
        "packet_count": len(messages),
        "packet_length_unique": sorted(set(int(x) for x in message_lengths.tolist())),
        "protocol": {"topic": "pose", "version": 4, "header_size": HEADER_SIZE, "fields": ["token_state", "frame_index", "left_hand_joints", "right_hand_joints"]},
        "round_trip_max_abs_error": max(round_trip_errors),
        "losses": losses,
        "prediction_stats": {
            "mean": float(pred.mean()),
            "std": float(pred.std()),
            "min": float(pred.min()),
            "max": float(pred.max()),
            "motion_token_l2_mean": float(np.linalg.norm(token, axis=1).mean()),
            "left_hand_abs_mean": float(np.abs(left).mean()),
            "right_hand_abs_mean": float(np.abs(right).mean()),
        },
        "output_npz": str(output_npz) if output_npz else None,
        "video": video_result,
        "zmq_publish": zmq_publish_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-mix", default="sonic_g1_smoke")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--video-backend", default="torchvision_av", choices=["decord", "opencv", "torchvision_av", "torchcodec"])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional action_head_final.pt checkpoint to load before prediction")
    parser.add_argument("--hidden-size", type=int, default=256, help="Action-head hidden size when no checkpoint metadata overrides it")
    parser.add_argument("--layers", type=int, default=1, help="Action-head transformer layer count when no checkpoint metadata overrides it")
    parser.add_argument("--inference-steps", type=int, default=2, help="Flow-matching inference steps for predict_action")
    parser.add_argument("--publish-zmq-host", default=None, help="Bind host for optional live ZMQ PUB, e.g. '*' or '127.0.0.1'")
    parser.add_argument("--publish-zmq-port", type=int, default=None, help="Port for optional live ZMQ PUB")
    parser.add_argument("--publish-rate-hz", type=float, default=50.0)
    parser.add_argument("--publish-warmup-sec", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    result = run_gate(
        data_root=args.data_root,
        data_mix=args.data_mix,
        steps=args.steps,
        batch_size=args.batch_size,
        device=device,
        video_backend=args.video_backend,
        output_npz=args.output_npz,
        output_video=args.output_video,
        checkpoint=args.checkpoint,
        hidden_size=args.hidden_size,
        layers=args.layers,
        inference_steps=args.inference_steps,
        publish_zmq_host=args.publish_zmq_host,
        publish_zmq_port=args.publish_zmq_port,
        publish_rate_hz=args.publish_rate_hz,
        publish_warmup_sec=args.publish_warmup_sec,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
