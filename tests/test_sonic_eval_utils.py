import json
from pathlib import Path

import numpy as np

from scripts.evaluate_sonic_real_token_checkpoint import (
    compute_action_metrics,
    export_sonic_action_stats,
    split_sonic_action,
    unnormalize_sonic_action,
)


def _write_dataset_stats(root: Path) -> Path:
    meta = root / "meta"
    meta.mkdir(parents=True)
    action = np.zeros((6, 78), dtype=np.float32)
    action[:, :64] = np.linspace(-1.0, 1.0, 6, dtype=np.float32)[:, None]
    action[:, 64:71] = 0.25
    action[:, 71:78] = -0.5
    stats = {
        "action": {
            "min": action.min(axis=0).tolist(),
            "max": action.max(axis=0).tolist(),
            "q01": np.quantile(action, 0.01, axis=0).tolist(),
            "q99": np.quantile(action, 0.99, axis=0).tolist(),
            "mean": action.mean(axis=0).tolist(),
            "std": action.std(axis=0).tolist(),
        }
    }
    path = meta / "stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    return path


def test_split_sonic_action_contract():
    action = np.zeros((2, 40, 78), dtype=np.float32)
    parts = split_sonic_action(action)
    assert parts["motion_token"].shape == (2, 40, 64)
    assert parts["left_hand_joints"].shape == (2, 40, 7)
    assert parts["right_hand_joints"].shape == (2, 40, 7)


def test_export_stats_preserves_key_slices_and_minmax(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset_stats(dataset)

    out = tmp_path / "sonic_action_stats.json"
    exported = export_sonic_action_stats(dataset, out, motion_mode="min_max")

    assert out.exists()
    assert exported["format"] == "sonic_action_stats_v1"
    assert exported["action.motion_token"]["start"] == 0
    assert exported["action.motion_token"]["end"] == 64
    assert exported["action.left_hand_joints"]["start"] == 64
    assert exported["action.right_hand_joints"]["end"] == 78
    assert len(exported["action.motion_token"]["min"]) == 64
    assert len(exported["action.left_hand_joints"]["min"]) == 7


def test_unnormalize_uses_minmax_and_constant_channels_clamp_to_constant(tmp_path):
    dataset = tmp_path / "dataset"
    _write_dataset_stats(dataset)
    stats = export_sonic_action_stats(dataset, tmp_path / "stats.json", motion_mode="min_max")

    normalized = np.ones((1, 40, 78), dtype=np.float32)
    raw = unnormalize_sonic_action(normalized, stats)

    assert raw.shape == (1, 40, 78)
    assert np.allclose(raw[..., :64], 1.0)
    assert np.allclose(raw[..., 64:71], 0.25)
    assert np.allclose(raw[..., 71:78], -0.5)


def test_metrics_report_per_slice_mse_range_and_smoothness():
    target = np.zeros((40, 78), dtype=np.float32)
    pred = np.zeros((40, 78), dtype=np.float32)
    pred[:, 0] = np.linspace(0.0, 1.0, 40)
    pred[:, 65] = 0.5

    metrics = compute_action_metrics(pred, target, motion_token_abs_limit=0.75)

    assert metrics["mse_total"] > 0
    assert metrics["mse_motion_token"] > 0
    assert metrics["mse_left_hand_joints"] > 0
    assert metrics["mse_right_hand_joints"] == 0
    assert metrics["motion_token_range_violation_rate"] > 0
    assert metrics["temporal_smoothness_l2_mean"] > 0
