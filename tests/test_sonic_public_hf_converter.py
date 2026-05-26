import numpy as np
import pytest

from scripts.convert_public_sonic_hf_lerobot import (
    EXPECTED_G1_WBC_NAMES,
    build_dit4dit_action,
    build_dit4dit_state,
    validate_unitree_g1_sonic_features,
)


def test_public_sonic_g1_mapping_extracts_body_state_and_hands():
    state43 = np.arange(43, dtype=np.float32)
    token64 = np.arange(100, 164, dtype=np.float32)
    wbc43 = np.arange(200, 243, dtype=np.float32)

    state29 = build_dit4dit_state(state43)
    action78 = build_dit4dit_action(token64, wbc43)

    assert state29.shape == (29,)
    assert state29.tolist() == list(range(22)) + list(range(29, 36))
    assert action78.shape == (78,)
    assert action78[:64].tolist() == token64.tolist()
    assert action78[64:71].tolist() == list(range(222, 229))
    assert action78[71:78].tolist() == list(range(236, 243))


def test_public_sonic_g1_mapping_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="observation.state"):
        build_dit4dit_state(np.zeros(42, dtype=np.float32))
    with pytest.raises(ValueError, match="action.motion_token"):
        build_dit4dit_action(np.zeros(63, dtype=np.float32), np.zeros(43, dtype=np.float32))
    with pytest.raises(ValueError, match="action.wbc"):
        build_dit4dit_action(np.zeros(64, dtype=np.float32), np.zeros(42, dtype=np.float32))


def test_validate_unitree_g1_sonic_features_requires_motion_token_wbc_and_g1_names():
    features = {
        "observation.images.ego_view": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.state": {"dtype": "float64", "shape": [43], "names": EXPECTED_G1_WBC_NAMES},
        "action.wbc": {"dtype": "float64", "shape": [43], "names": EXPECTED_G1_WBC_NAMES},
        "action.motion_token": {"dtype": "float64", "shape": [64]},
    }
    validate_unitree_g1_sonic_features(features)

    bad = dict(features)
    bad["action.wbc"] = {"dtype": "float64", "shape": [43], "names": list(reversed(EXPECTED_G1_WBC_NAMES))}
    with pytest.raises(ValueError, match="Unitree G1 joint names"):
        validate_unitree_g1_sonic_features(bad)
