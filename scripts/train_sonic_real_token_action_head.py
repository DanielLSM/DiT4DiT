#!/usr/bin/env python3
"""Train DiT4DiT's FlowmatchingActionHead on a SONIC real-token LeRobot dataset.

This is intentionally a pilot trainer: it exercises the DiT4DiT LeRobot dataloader
and the real action-head loss/inference path while replacing the heavyweight VLM
backbone with deterministic synthetic language/vision embeddings. Use it to finish
an end-to-end motion-token training run before scaling to the full Cosmos backbone.
"""

from __future__ import annotations

import argparse
import json
import random
import time
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


def _build_action_head(action_dim: int, state_dim: int, horizon: int, hidden_size: int, layers: int, device: torch.device) -> FlowmatchingActionHead:
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
                    "num_inference_timesteps": 4,
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


def _collate(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.stack([s["action"] for s in samples]).astype(np.float32)
    states = np.stack([s["state"] for s in samples]).astype(np.float32)
    masks = np.stack([s["action_mask"] for s in samples]).astype(np.float32)
    return actions, states, masks


def _sample_batch(dataset: Any, batch_size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idxs = [rng.randrange(len(dataset)) for _ in range(batch_size)]
    return _collate([dataset[i] for i in idxs])


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    data_cfg = OmegaConf.create(
        {
            "data_root_dir": str(args.data_root),
            "data_mix": args.data_mix,
            "delete_pause_frame": False,
            "lerobot_version": "v2.0",
            "video_backend": args.video_backend,
            "video_delta_indices": [0, 1],
            "max_state_dim": 29,
            "max_action_dim": 78,
            "action_video_freq_ratio": 1,
        }
    )
    dataset = get_vla_dataset(data_cfg=data_cfg, mode="train", seed=args.seed)
    if len(dataset) <= 0:
        raise AssertionError("dataset has no samples")
    actions, states, masks = _sample_batch(dataset, min(args.batch_size, len(dataset)), rng)
    if actions.shape[-1] != 78 or states.shape[-1] != 29:
        raise AssertionError(f"bad shapes: actions={actions.shape}, states={states.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise AssertionError("dataset contains non-finite action/state")

    horizon = int(actions.shape[1])
    model = _build_action_head(78, 29, horizon, args.hidden_size, args.layers, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    torch.manual_seed(args.seed + 1000)
    vl_embs_pool = torch.randn((max(args.batch_size, 1), args.vl_tokens, 768), device=device, dtype=torch.float32)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    losses: list[float] = []
    t0 = time.time()
    model.train()
    with log_path.open("w", encoding="utf-8") as log_f:
        for step in range(1, args.steps + 1):
            ba, bs, bm = _sample_batch(dataset, args.batch_size, rng)
            action_t = torch.as_tensor(ba, device=device, dtype=torch.float32)
            state_t = torch.as_tensor(bs, device=device, dtype=torch.float32)
            mask_t = torch.as_tensor(bm, device=device, dtype=torch.float32)
            if vl_embs_pool.shape[0] != args.batch_size:
                vl_embs = torch.randn((args.batch_size, args.vl_tokens, 768), device=device, dtype=torch.float32)
            else:
                vl_embs = vl_embs_pool
            optimizer.zero_grad(set_to_none=True)
            loss = model(vl_embs, action_t, mask_t, state_t)
            if not torch.isfinite(loss):
                raise AssertionError(f"non-finite loss at step {step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_f = float(loss.detach().cpu())
            losses.append(loss_f)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                rec = {"step": step, "loss": loss_f, "elapsed_sec": time.time() - t0, "lr": args.lr}
                print(json.dumps(rec, sort_keys=True), flush=True)
                log_f.write(json.dumps(rec, sort_keys=True) + "\n")
                log_f.flush()

    model.eval()
    eval_actions, eval_states, _ = _sample_batch(dataset, 1, rng)
    with torch.no_grad():
        pred = model.predict_action(
            torch.randn((1, args.vl_tokens, 768), device=device, dtype=torch.float32),
            torch.as_tensor(eval_states, device=device, dtype=torch.float32),
        )
    if tuple(pred.shape) != (1, horizon, 78):
        raise AssertionError(f"bad prediction shape: {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise AssertionError("prediction contains non-finite values")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "losses": losses,
        "dataset_len": len(dataset),
        "horizon": horizon,
        "action_dim": 78,
        "state_dim": 29,
    }
    ckpt_path = output_dir / "action_head_final.pt"
    torch.save(ckpt, ckpt_path)
    result = {
        "ok": True,
        "dataset_len": len(dataset),
        "data_root": str(args.data_root),
        "data_mix": args.data_mix,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "horizon": horizon,
        "loss_first": losses[0],
        "loss_final": losses[-1],
        "loss_min": min(losses),
        "loss_mean_first_10": float(np.mean(losses[: min(10, len(losses))])),
        "loss_mean_last_10": float(np.mean(losses[max(0, len(losses) - 10) :])),
        "loss_delta": losses[-1] - losses[0],
        "prediction_shape": list(pred.shape),
        "prediction_abs_mean": float(pred.abs().mean().detach().cpu()),
        "target_token_abs_mean": float(np.mean(np.abs(eval_actions[..., :64]))),
        "target_hand_abs_mean": float(np.mean(np.abs(eval_actions[..., 64:]))),
        "checkpoint": str(ckpt_path),
        "log_path": str(log_path),
        "elapsed_sec": time.time() - t0,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--data-mix", default="sonic_g1_real_token")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--vl-tokens", type=int, default=8)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--video-backend", default="torchvision_av", choices=["decord", "opencv", "torchvision_av", "torchcodec"])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
