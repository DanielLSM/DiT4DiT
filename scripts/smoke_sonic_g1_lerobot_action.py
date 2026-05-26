#!/usr/bin/env python3
"""Smoke-test the SONIC G1 LeRobot fixture through DiT4DiT's dataloader and action head.

This deliberately avoids the heavy Cosmos backbone: it verifies that the tiny fixture
loads through the repository's LeRobot mixture path, produces G1 state + 78D action
batches, then runs a real FlowmatchingActionHead train/eval loop on CUDA when
available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset
from DiT4DiT.model.modules.action_model.ActionDiT import FlowmatchingActionHead


def _ns(mapping: dict[str, Any]) -> SimpleNamespace:
    out = SimpleNamespace()
    for key, value in mapping.items():
        setattr(out, key, _ns(value) if isinstance(value, dict) else value)
    return out


def _build_action_head(action_dim: int, state_dim: int, horizon: int, device: torch.device) -> FlowmatchingActionHead:
    cfg = _ns(
        {
            "framework": {
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 256,
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
                    "num_inference_timesteps": 2,
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
        "num_layers": 1,
        "output_dim": 256,
        "positional_embeddings": None,
    }
    return FlowmatchingActionHead(cfg).to(device)


def run_smoke(
    data_root: Path,
    data_mix: str,
    steps: int,
    batch_size: int,
    device: torch.device,
    video_backend: str,
) -> dict[str, Any]:
    data_cfg = OmegaConf.create(
        {
            "data_root_dir": str(data_root),
            "data_mix": data_mix,
            "delete_pause_frame": False,
            "lerobot_version": "v2.0",
            "video_backend": video_backend,
            "video_delta_indices": [0, 1],
            "max_state_dim": 29,
            "max_action_dim": 78,
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
    image_lengths = [len(s["image"]) for s in samples]

    if actions.shape[-1] != 78:
        raise AssertionError(f"expected 78D action, got {actions.shape}")
    if states.shape[-1] != 29:
        raise AssertionError(f"expected 29D G1 state, got {states.shape}")
    if actions.shape[1] < 2:
        raise AssertionError(f"expected action horizon >=2, got {actions.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise AssertionError("actions/states contain non-finite values")
    if not all(n >= 1 for n in image_lengths):
        raise AssertionError(f"missing decoded images: {image_lengths}")

    horizon = int(actions.shape[1])
    action_dim = int(actions.shape[-1])
    state_dim = int(states.shape[-1])
    model = _build_action_head(action_dim=action_dim, state_dim=state_dim, horizon=horizon, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    action_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
    state_t = torch.as_tensor(states, device=device, dtype=torch.float32)
    mask_t = torch.as_tensor(masks, device=device, dtype=torch.float32)
    # Synthetic VLM tokens: intentionally small, fixed width required by DiT-B.
    torch.manual_seed(123)
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
    pred = model.predict_action(vl_embs[:1], state_t[:1])
    if tuple(pred.shape) != (1, horizon, action_dim):
        raise AssertionError(f"bad prediction shape {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise AssertionError("prediction contains non-finite values")

    return {
        "data_root": str(data_root),
        "data_mix": data_mix,
        "dataset_len": len(dataset),
        "batch_size": batch_size,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "action_shape": list(actions.shape),
        "state_shape": list(states.shape),
        "image_lengths": image_lengths,
        "losses": losses,
        "final_loss": losses[-1],
        "prediction_shape": list(pred.shape),
        "prediction_abs_mean": float(pred.abs().mean().detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Parent directory containing sonic_g1_sample_lerobot")
    parser.add_argument("--data-mix", default="sonic_g1_smoke")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--video-backend", default="decord", choices=["decord", "opencv", "torchvision_av", "torchcodec"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    result = run_smoke(
        data_root=args.data_root,
        data_mix=args.data_mix,
        steps=args.steps,
        batch_size=args.batch_size,
        device=device,
        video_backend=args.video_backend,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
