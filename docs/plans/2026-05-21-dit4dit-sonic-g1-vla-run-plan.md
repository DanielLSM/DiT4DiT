# DiT4DiT SONIC G1 VLA Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement, train, evaluate, and eventually run DiT4DiT as a SONIC-compatible VLA backend for a Unitree G1 robot by predicting SONIC's continuous 78D latent action interface.

**Architecture:** Keep DiT4DiT as the model/training repo and GR00T-WholeBodyControl/SONIC as the robot-control repo. First prove the DiT4DiT model can consume SONIC/Unitree G1 data and predict `[H, 78]` action chunks offline; only after that add a thin SONIC-side policy adapter that reuses existing SONIC safety, range-checking, and publishing paths.

**Tech Stack:** DiT4DiT, Cosmos-Predict2.5-2B, PyTorch 2.7/CUDA 12.8, Accelerate/DeepSpeed, LeRobot-style datasets, GR00T-WholeBodyControl/SONIC VLA export, Unitree G1 deployment stack.

---

## Current branch and scope

- DiT4DiT worktree: `/home/daniel/dev/dit4dit/sonic-token-actions`
- Branch: `experiment/sonic-token-actions`
- Current upstream base: `0362b34` (`fix mul wrist`)
- Current branch commit: `ba3a661` (`docs: plan SONIC token adaptation`)
- Remote layout:
  - `origin = https://github.com/Mondo-Robotics/DiT4DiT.git`, push disabled
  - `fork = git@github.com:DanielLSM/DiT4DiT.git`

This plan belongs in DiT4DiT because the first unknown is model/data compatibility, not robot publishing. Do not start by making a large SONIC-side branch. That would be the robotics equivalent of operating before doing imaging.

## Evidence from this repo

### DiT4DiT has partial real Unitree G1 scaffolding

Relevant files:

- `DiT4DiT/config/real_robot/dit4dit_g1.yaml`
- `examples/Real_G1/train_files/run_real_robot.sh`
- `examples/Real_G1/eval_files/eval_policy.py`
- `examples/Real_G1/eval_files/eval_real_world.sh`
- `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`
- `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`

Current real-G1 config is not SONIC-compatible yet:

- `framework.action_model.action_dim: 32`
- `framework.action_model.action_horizon: 16`
- `framework.action_model.future_action_window_size: 15`
- `datasets.vla_data.max_action_dim: 32`
- `datasets.vla_data.action_type: joint_position`
- `datasets.vla_data.data_mix: real_robot_all`

So it is a real-robot joint-position path, not a SONIC latent-token path.

### Current Unitree G1 data config is arms/grippers, not SONIC

`UnitreeG1DataConfig` and `UnitreeG1AlohaOnlyArmsDataConfig` exist in `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`.

The registered G1 config is:

```python
"g1_body29_aloha_arms_only": UnitreeG1AlohaOnlyArmsDataConfig()
```

Its keys are:

```text
video.ego_view
state.left_arm
state.right_arm
state.left_gripper
state.right_gripper
action.left_arm
action.right_arm
action.left_gripper
action.right_gripper
annotation.human.task_description
```

Missing for SONIC:

```text
action.motion_token
action.left_hand_joints
action.right_hand_joints
```

Therefore the correct implementation is not to reuse `g1_body29_aloha_arms_only` directly. It should be copied conceptually and replaced with a new SONIC action schema.

### Data pointers found in the repo

Public/reproducible pointers:

1. Backbone:

```bash
huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --local-dir /path/outside/home/Cosmos-Predict2.5-2B
```

2. Public DiT4DiT checkpoints:

```bash
huggingface-cli download mondo-robotics/dit4dit-model \
  --include "dit4dit_libero/*" \
  --local-dir /path/outside/home/dit4dit-model

huggingface-cli download mondo-robotics/dit4dit-model \
  --include "dit4dit_robocasa_gr1/*" \
  --local-dir /path/outside/home/dit4dit-model
```

3. Public GR00T-X/Fourier GR1 simulation dataset pointer from repo docs:

```bash
python examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py
```

This downloads 24 task folders from:

```text
nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
```

into the repo's default `./playground/...` path. Do not use that default for a real run because it writes generated data under the git checkout/home. Patch the script or pass an external destination.

Private/unresolved pointers:

`DATASET_NAMED_MIXTURES["real_robot_all"]` points at seven real-robot dataset folder names:

```text
pnp_eggplant_lh_200ep_26_1_28
pnp_corn_middle_drawer
pnp_plate
arrange_flower
move_spoon
stack_cups
packaging
```

All use robot type:

```text
g1_body29_aloha_arms_only
```

The repo does not provide a public download script or Hugging Face URL for these real Unitree G1 folders. README also says real robot quick start is "Coming soon" and the TODO list still includes releasing Unitree G1 tabletop and whole-body code. So: the repo contains useful names/configs, but not enough released data to train a Unitree G1 SONIC policy by itself.

### Data conclusion

For SONIC G1, usable training data must come from one of these sources:

1. **Best:** SONIC/GR00T VLA exported trajectories containing `action.motion_token`, hand joints, video, state, and language.
2. **Possible but not enough:** DiT4DiT private-style real G1 folders if we can obtain them from the authors; they are joint-position/ALOHA-arms data and still need conversion or relabeling into SONIC latent actions.
3. **Useful for plumbing only:** public GR00T-X/Fourier GR1 simulation data; not Unitree G1 and not SONIC latent tokens.
4. **Fallback:** generate a small SONIC dataset ourselves via the existing GR00T-WholeBodyControl exporter.

## GR00T/SONIC data bridge

Yes: GR00T-WholeBodyControl is the right place to source data for this DiT4DiT branch.

The relevant GR00T/SONIC docs and code establish an end-to-end collection path:

- `docs/source/tutorials/vla_workflow.md` says the VLA workflow is collect teleop demos, fine-tune, then deploy, and explicitly defines the VLA action as 64D SONIC latent token plus 7D left hand plus 7D right hand.
- `docs/source/tutorials/data_collection.md` says the exporter records LeRobot v2.1 datasets with `data/`, `videos/`, and `meta/` directories, and supports `--root-output-dir`.
- `gear_sonic/data/features_sonic_vla.py` defines the actual modality mapping:
  - video: `observation.images.ego_view` -> `video.ego_view`
  - language: `task_index` -> `annotation.human.task_description`
  - action: `action.motion_token` -> `action.motion_token`, 64D
  - action: `teleop.left_hand_joints` -> `action.left_hand_joints`, 7D
  - action: `teleop.right_hand_joints` -> `action.right_hand_joints`, 7D
- `gear_sonic/scripts/run_data_exporter.py` writes `action.motion_token` from C++/proprio `token_state`, and hand actions from the current teleop/planner message.

That is better than trying to infer SONIC labels from DiT4DiT's existing G1 folders. DiT4DiT's `real_robot_all` is useful evidence that the repo has a real-G1 loader style, but those folders are not publicly available here and target joint-position/ALOHA-arm data, not SONIC latents.

Preferred data route:

```text
GR00T-WholeBodyControl data exporter
  -> LeRobot v2.1 SONIC dataset outside $HOME
  -> DiT4DiT UnitreeG1SonicTokenDataConfig
  -> train/evaluate [B, 40, 78]
```

Example collection command, with storage fixed to a non-home path:

```bash
cd /home/daniel/dev/groot-wholebodycontrol/sonic-vla-pipeline
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "pick up the cup" \
  --dataset-name sonic_g1_pick_cup \
  --root-output-dir /cluster/scratch/$USER/sonic-vla-data \
  --camera-host 192.168.123.164 \
  --camera-port 5555
```

Or all-in-one real-robot collection, if the launcher is patched/extended to pass `--root-output-dir` through to the exporter:

```bash
python gear_sonic/scripts/launch_data_collection.py \
  --camera-host 192.168.123.164 \
  --task-prompt "pick up the cup" \
  --dataset-name sonic_g1_pick_cup
```

Important implementation note: `launch_data_collection.py` currently exposes `dataset_name` but not `root_output_dir` in its dataclass, while `run_data_exporter.py` has `root_output_dir`. For serious data collection, either run `run_data_exporter.py` manually with `--root-output-dir`, or add a small launcher option so large datasets do not land under the repo/home.

## SONIC action contract

The target action output is continuous, not categorical token IDs.

Per action step:

```text
action.motion_token:        64 floats in SONIC latent token space
action.left_hand_joints:     7 floats
action.right_hand_joints:    7 floats
total:                      78 floats
```

Canonical flat layout inside DiT4DiT:

```python
action[..., 0:64] = action.motion_token
action[..., 64:71] = action.left_hand_joints
action[..., 71:78] = action.right_hand_joints
```

Target deployment horizon:

```text
H = 40
final model output shape = [B, 40, 78]
```

A shorter `H=16` debug config is acceptable only for smoke tests. It must not be confused with the final SONIC deployment contract.


## Safety rule for real Unitree G1

Do not run on the real robot until these gates pass:

1. Dataloader emits `[B, 40, 78]` with the expected key ordering.
2. Tiny overfit on 10-50 trajectories reduces loss clearly.
3. Held-out offline metrics are sane per slice: motion-token, left hand, right hand.
4. Predicted motion-token values stay within the expected SONIC latent range used by the existing inference bridge.
5. Temporal smoothness is not pathological.
6. Decoder/replay or sim test passes.
7. Inference latency supports the SONIC client loop.
8. Robot-side adapter has an e-stop, range clipping, rate limiting, and a dry-run mode.

Shape compatibility is not robot compatibility. `[40, 78]` can still be a very elegant way to fall over.

---

## Phase 1: Pin data availability and one real sample

**Objective:** Determine whether we have real SONIC/Unitree G1 training data and exactly what keys it exposes.

**Files:**

- Read: `/home/daniel/dev/groot-wholebodycontrol/sonic-vla-pipeline/gear_sonic/data/features_sonic_vla.py`
- Read: `/home/daniel/dev/groot-wholebodycontrol/sonic-vla-pipeline/gear_sonic/scripts/run_data_exporter.py`
- Create: `tools/sonic/inspect_sonic_lerobot_dataset.py`
- Create: `docs/sonic_g1_data_inventory.md`

**Step 1: Inventory DiT4DiT pointers**

Run:

```bash
cd /home/daniel/dev/dit4dit/sonic-token-actions
PYTHONPATH=. python - <<'PY'
from DiT4DiT.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
for name in ["real_robot_all", "fourier_gr1_unified_1000"]:
    print(f"## {name}")
    for item in DATASET_NAMED_MIXTURES[name]:
        print(item)
PY
```

Expected:

- `real_robot_all` lists the seven real G1 folder names above.
- `fourier_gr1_unified_1000` lists public Fourier GR1 simulation task folders.

**Step 2: Locate or create SONIC export data**

Do not store data under `/home/daniel` or inside the repo. Use a target like:

```bash
export SONIC_DATA_ROOT=/cluster/scratch/$USER/dit4dit-data/sonic-g1-lerobot
export DIT4DIT_RUN_ROOT=/cluster/scratch/$USER/dit4dit-runs/sonic-g1
```

If running on a non-Clariden machine, replace with the appropriate non-home scratch/project/data path.

**Step 3: Inspect one sample**

The inspection script must print:

- dataset path
- number of episodes
- LeRobot version
- video keys and shapes
- state keys and shapes
- language key
- action keys and dimensions
- min/max/mean/std for `action.motion_token`
- min/max/mean/std for both hand-joint fields
- whether every episode has at least 40 future action steps

**Verification command:**

```bash
PYTHONPATH=. python tools/sonic/inspect_sonic_lerobot_dataset.py \
  --dataset-root "$SONIC_DATA_ROOT" \
  --limit-episodes 3
```

Expected: script exits 0 and writes `docs/sonic_g1_data_inventory.md` with exact observed keys.

---

## Phase 2: Add a SONIC action schema helper

**Objective:** Make the 78D contract explicit and testable before changing the loader.

**Files:**

- Create: `DiT4DiT/dataloader/gr00t_lerobot/sonic_action_schema.py`
- Create: `tests/test_sonic_action_schema.py`

**Step 1: Add constants and split/concat helpers**

Implement:

```python
MOTION_TOKEN_DIM = 64
LEFT_HAND_DIM = 7
RIGHT_HAND_DIM = 7
ACTION_DIM = MOTION_TOKEN_DIM + LEFT_HAND_DIM + RIGHT_HAND_DIM
TARGET_HORIZON = 40

MOTION_SLICE = slice(0, 64)
LEFT_HAND_SLICE = slice(64, 71)
RIGHT_HAND_SLICE = slice(71, 78)
```

Functions:

```python
def split_sonic_action(action):
    """Accept [..., 78], return dict with motion_token, left_hand_joints, right_hand_joints."""


def concat_sonic_action(motion_token, left_hand_joints, right_hand_joints):
    """Return [..., 78] with canonical SONIC ordering."""
```

**Step 2: Unit tests**

Test shapes:

- `[78]`
- `[40, 78]`
- `[B, 40, 78]`

Also test that wrong last dimension fails loudly.

**Verification:**

```bash
cd /home/daniel/dev/dit4dit/sonic-token-actions
PYTHONPATH=. pytest tests/test_sonic_action_schema.py -q
```

Expected: all tests pass without CUDA, Cosmos, or robot data.

---

## Phase 3: Add a SONIC G1 data config

**Objective:** Teach DiT4DiT to load the SONIC VLA dataset with the correct keys and action horizon.

**Files:**

- Modify: `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`
- Modify: `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`
- Modify: `DiT4DiT/dataloader/gr00t_lerobot/embodiment_tags.py` only if needed
- Create: `tests/test_sonic_data_config.py`

**Step 1: Add `UnitreeG1SonicTokenDataConfig`**

Start from `UnitreeG1AlohaOnlyArmsDataConfig`, but use SONIC action keys:

```python
class UnitreeG1SonicTokenDataConfig(BaseDataConfig):
    video_keys = ["video.<observed_camera_key>"]
    state_keys = [
        # Fill from the inspected SONIC export. Do not invent these.
    ]
    action_keys = [
        "action.motion_token",
        "action.left_hand_joints",
        "action.right_hand_joints",
    ]
    language_keys = ["annotation.<observed_language_key>"]
    observation_indices = [0]
    action_indices = list(range(40))
```

Use observed keys from Phase 1. If the SONIC exporter uses `video.image`, `video.ego_view`, or `video.rs_view`, record the exact one in the data inventory.

**Step 2: Normalize all SONIC action fields**

Initial normalization modes:

```python
StateActionTransform(
    apply_to=self.action_keys,
    normalization_modes={
        "action.motion_token": "min_max",  # or q99 if observed outliers make min/max brittle
        "action.left_hand_joints": "min_max",
        "action.right_hand_joints": "min_max",
    },
)
```

Decision rule:

- Use `min_max` if the exporter stats are bounded and stable.
- Use `q99` if outliers distort min/max.
- Do not use `binary` for hand joints; they are continuous 7D values.

**Step 3: Register config and mixture**

Add:

```python
ROBOT_TYPE_CONFIG_MAP["unitree_g1_sonic_token"] = UnitreeG1SonicTokenDataConfig()
```

Add a mixture:

```python
"sonic_g1_token_all": [
    ("<dataset-folder-name>", 1.0, "unitree_g1_sonic_token"),
]
```

If multiple tasks exist, list each folder explicitly. Avoid globbing in the mixture registry; it hides missing data.

**Step 4: Embodiment tag**

Use `NEW_EMBODIMENT` first unless there is a strong reason to add a new embedding ID. The action semantics are new; pretending it is the existing GR1 or G1 arm-joint embodiment is a nice way to get misleading results.

**Verification:**

```bash
PYTHONPATH=. pytest tests/test_sonic_data_config.py -q
PYTHONPATH=. python tools/sonic/inspect_sonic_lerobot_dataset.py \
  --dataset-root "$SONIC_DATA_ROOT" \
  --robot-type unitree_g1_sonic_token
```

Expected:

- Data config exists in `ROBOT_TYPE_CONFIG_MAP`.
- Mixture exists in `DATASET_NAMED_MIXTURES`.
- Loaded sample contains a normalized action tensor with last dimension 78.

---

## Phase 4: Add the DiT4DiT SONIC G1 training config

**Objective:** Create a config that trains DiT4DiT on SONIC 78D/40-step actions.

**Files:**

- Create: `DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml`
- Create: `examples/Real_G1/train_files/run_sonic_g1.sh`
- Create: `examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh`

**Config base:** copy `DiT4DiT/config/real_robot/dit4dit_g1.yaml` and change only what is required.

Required changes:

```yaml
framework:
  action_model:
    action_dim: 78
    state_dim: 64
    future_action_window_size: 39
    action_horizon: 40

datasets:
  vla_data:
    data_root_dir: /path/outside/home/to/sonic-g1-lerobot
    data_mix: sonic_g1_token_all
    action_type: sonic_latent_token_plus_hands
    max_action_dim: 78
    # Confirm from data inventory before freezing this:
    action_video_freq_ratio: 1
```

Initial trainer settings for smoke/overfit:

```yaml
trainer:
  max_train_steps: 500
  save_interval: 250
  eval_interval: 50
  logging_frequency: 10
  freeze_modules: "backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"
```

Full training can restore larger step counts after the overfit gate passes.

**Storage rule:** default `run_root_dir: results/Checkpoints` must be overridden by the shell script to a non-home scratch path.

Example script variables:

```bash
export DATA_ROOT=/cluster/scratch/$USER/dit4dit-data/sonic-g1-lerobot
export RUN_ROOT=/cluster/scratch/$USER/dit4dit-runs/sonic-g1
export COSMOS_ROOT=/cluster/scratch/$USER/models/Cosmos-Predict2.5-2B
```

**Verification:**

```bash
PYTHONPATH=. python - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load("DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml")
assert cfg.framework.action_model.action_dim == 78
assert cfg.framework.action_model.action_horizon == 40
assert cfg.framework.action_model.future_action_window_size == 39
assert cfg.datasets.vla_data.max_action_dim == 78
assert cfg.datasets.vla_data.data_mix == "sonic_g1_token_all"
print("sonic_g1 config ok")
PY
```

---

## Phase 5: Dataloader and one-batch smoke test

**Objective:** Prove the DiT4DiT loader path produces the correct tensors before importing the full model.

**Files:**

- Create: `tools/sonic/smoke_sonic_dataloader.py`
- Create: `tests/test_sonic_dataset_contract.py`

**Script requirements:**

`tools/sonic/smoke_sonic_dataloader.py` must:

- load `dit4dit_sonic_g1.yaml`
- instantiate the dataset through `get_vla_dataset`
- retrieve one sample and one DataLoader batch
- print keys, shapes, dtypes
- assert action last dimension is 78
- assert target horizon is 40
- print per-slice stats using `split_sonic_action`

**Verification:**

```bash
PYTHONPATH=. python tools/sonic/smoke_sonic_dataloader.py \
  --config DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml \
  --data-root "$SONIC_DATA_ROOT" \
  --data-mix sonic_g1_token_all \
  --num-workers 0
```

Expected output includes:

```text
action shape: [40, 78]
motion_token shape: [40, 64]
left_hand_joints shape: [40, 7]
right_hand_joints shape: [40, 7]
```

---

## Phase 6: Model forward/predict smoke test

**Objective:** Verify DiT4DiT can construct and sample a `[B, 40, 78]` action trajectory.

**Files:**

- Create: `tools/sonic/smoke_sonic_model.py`
- Modify: `deployment/model_server/server_policy.py` only if needed
- Modify: `deployment/model_server/tools/websocket_policy_server.py` only if needed

**Step 1: Local model import**

Run with a tiny config and real Cosmos path:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/sonic/smoke_sonic_model.py \
  --config DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml \
  --cosmos-root "$COSMOS_ROOT" \
  --data-root "$SONIC_DATA_ROOT" \
  --data-mix sonic_g1_token_all
```

Expected:

- model loads
- one batch forward returns `action_loss`
- `predict_action([example])` returns `normalized_actions` with shape `[1, 40, 78]`

**Step 2: WebSocket API sanity**

Existing server currently calls:

```python
self._policy.predict_action(**msg)
```

That works only if the client sends a flat message containing `examples=...`. The server docstring mentions a structured `payload`, but it does not actually extract `payload`. For SONIC deployment, make this boring and explicit:

- accept `{"type": "infer", "examples": [...]}`
- or patch the route to extract `payload = msg.get("payload", msg)` and call `predict_action(**payload)`
- add a tiny test for both message shapes

Do not debug this on the robot. Network protocol bugs belong in a unit test, not near ankles.

---

## Phase 7: Tiny overfit

**Objective:** Confirm the model can learn the SONIC action labels on a tiny fixed subset.

**Files:**

- Create: `examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh`
- Create: `tools/sonic/evaluate_sonic_predictions.py`

**Run:**

```bash
cd /home/daniel/dev/dit4dit/sonic-token-actions
bash examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh
```

Suggested pilot budget:

- 10-50 trajectories
- 1 GPU if feasible, otherwise smallest multi-GPU run that fits
- 500-2,000 training steps
- frozen text encoder and VAE initially
- outputs to `$RUN_ROOT/tiny-overfit-*`, not under the repo

**Metrics:**

- train action loss curve
- total action MSE
- motion-token MSE
- left-hand MSE
- right-hand MSE
- max absolute motion-token value
- percentage outside SONIC expected bound
- temporal smoothness: mean `||a[t+1] - a[t]||`

**Pass gate:**

- training loss drops substantially on the tiny subset
- per-slice MSE decreases
- predicted motion-token range is not exploding
- predicted trajectories are temporally smooth enough to replay

---

## Phase 8: Held-out offline evaluation

**Objective:** Decide whether the model is good enough to justify robot-side integration.

**Files:**

- Create: `examples/Real_G1/eval_files/eval_sonic_g1_offline.py`
- Create: `tools/sonic/evaluate_sonic_predictions.py`
- Create: `docs/reports/sonic_g1_offline_eval_template.md`

**Run:**

```bash
PYTHONPATH=. python examples/Real_G1/eval_files/eval_sonic_g1_offline.py \
  --model_path "$RUN_ROOT/<checkpoint>/pytorch_model.pt" \
  --dataset_root "$SONIC_DATA_ROOT" \
  --data_config unitree_g1_sonic_token \
  --data_mix sonic_g1_token_all \
  --action_horizon 40 \
  --save_dir "$RUN_ROOT/offline-eval/<run-id>"
```

**Report fields:**

- checkpoint path
- dataset root
- held-out tasks/episodes
- action shape
- per-slice MSE
- per-slice normalized MSE
- range-violation rate
- smoothness metrics
- example trajectory plots
- inference latency per chunk

**Decision gate before robot-side branch:**

Proceed only if held-out metrics are stable and the predictions look plausible under SONIC decoder/replay. If the 64D latent slice drifts outside the logged distribution, stop. The robot has enough degrees of freedom; it does not need our imagination added.

---

## Phase 9: Add a SONIC-side DiT4DiT policy adapter

**Objective:** Connect a trained DiT4DiT policy server to SONIC while reusing existing SONIC deployment code.

**Repo:** this phase should be in GR00T-WholeBodyControl, not DiT4DiT.

Suggested worktree:

```text
/home/daniel/dev/groot-wholebodycontrol/dit4dit-policy-client
branch: experiment/dit4dit-policy-client
```

**Files likely touched in GR00T-WholeBodyControl:**

- Read/copy patterns from: `gear_sonic/scripts/run_vla_inference.py`
- Create: `gear_sonic/scripts/run_dit4dit_inference.py`
- Create: `gear_sonic/utils/inference/dit4dit_client.py`
- Add tests/mocks if test framework exists

**Adapter behavior:**

1. Collect camera/state/language exactly like existing SONIC VLA inference.
2. Build DiT4DiT example:

```python
example = {
    "image": [pil_or_numpy_image],
    "lang": instruction,
    "state": normalized_state_or_none,
}
```

3. Call DiT4DiT server.
4. Receive `normalized_actions: [1, 40, 78]`.
5. Unnormalize using the training checkpoint's stats.
6. Split into:

```python
motion_token = action[:, 0:64]
left_hand_joints = action[:, 64:71]
right_hand_joints = action[:, 71:78]
```

7. Reuse existing SONIC packing/publish logic.
8. Preserve existing range check for `motion_token`.
9. Publish action chunks at the same control boundary as existing SONIC VLA path.

**Important:** If DiT4DiT uses its own normalization stats, the SONIC-side adapter must load exactly those stats from the checkpoint. Do not hand-scale the latent vector by vibes. Vibes have poor unit tests.

---

## Phase 10: Real Unitree G1 run ladder

**Objective:** Reach real robot only after the software and action distribution have been de-risked.

**Run ladder:**

1. Unit test: schema split/concat.
2. Dataset test: one real sample emits `[40, 78]`.
3. Model test: `predict_action` emits `[1, 40, 78]`.
4. Server test: WebSocket request/response works locally.
5. Offline replay: compare predictions against logged SONIC actions.
6. Decoder-only test: feed predicted chunks into SONIC decoder/replay without motors.
7. Robot dry-run: live camera/state, inference only, no action publish.
8. Robot shadow mode: publish to log only while native controller runs.
9. Robot low-risk mode: constrained task, reduced speed/range, hands disabled if useful.
10. Full task attempt with human operator and e-stop.

**Robot-run checklist:**

- e-stop tested
- action publish disabled by default
- explicit `--enable-publish` flag required
- max motion-token absolute value guard enabled
- hand joint limits enabled
- stale-action timeout enabled
- server disconnect behavior tested
- logs written outside home
- exact checkpoint SHA/path recorded
- exact dataset/run config recorded

---

## Implementation task sequence

### Task 1: Data inventory note

**Objective:** Record what data is actually available before writing adapters.

**Files:**

- Create: `docs/sonic_g1_data_inventory.md`
- Create: `tools/sonic/inspect_sonic_lerobot_dataset.py`

**Verify:**

```bash
PYTHONPATH=. python tools/sonic/inspect_sonic_lerobot_dataset.py --help
```

Commit message:

```bash
git add docs/sonic_g1_data_inventory.md tools/sonic/inspect_sonic_lerobot_dataset.py
git commit -m "docs: inventory SONIC G1 data contract"
```

### Task 2: SONIC schema helper

**Objective:** Add tested split/concat helpers.

**Files:**

- Create: `DiT4DiT/dataloader/gr00t_lerobot/sonic_action_schema.py`
- Create: `tests/test_sonic_action_schema.py`

**Verify:**

```bash
PYTHONPATH=. pytest tests/test_sonic_action_schema.py -q
```

Commit message:

```bash
git add DiT4DiT/dataloader/gr00t_lerobot/sonic_action_schema.py tests/test_sonic_action_schema.py
git commit -m "feat: add SONIC action schema helpers"
```

### Task 3: SONIC data config and mixture

**Objective:** Register `unitree_g1_sonic_token` and `sonic_g1_token_all`.

**Files:**

- Modify: `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`
- Modify: `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`
- Create: `tests/test_sonic_data_config.py`

**Verify:**

```bash
PYTHONPATH=. pytest tests/test_sonic_data_config.py -q
```

Commit message:

```bash
git add DiT4DiT/dataloader/gr00t_lerobot/data_config.py DiT4DiT/dataloader/gr00t_lerobot/mixtures.py tests/test_sonic_data_config.py
git commit -m "feat: register SONIC G1 data config"
```

### Task 4: SONIC training config and scripts

**Objective:** Add `dit4dit_sonic_g1.yaml` and safe run scripts.

**Files:**

- Create: `DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml`
- Create: `examples/Real_G1/train_files/run_sonic_g1.sh`
- Create: `examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh`

**Verify:**

```bash
PYTHONPATH=. python - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load("DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml")
assert cfg.framework.action_model.action_dim == 78
assert cfg.framework.action_model.action_horizon == 40
assert cfg.datasets.vla_data.max_action_dim == 78
print("config ok")
PY
bash -n examples/Real_G1/train_files/run_sonic_g1.sh
bash -n examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh
```

Commit message:

```bash
git add DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml examples/Real_G1/train_files/run_sonic_g1*.sh
git commit -m "feat: add SONIC G1 training config"
```

### Task 5: Smoke scripts

**Objective:** Validate dataloader and model shape path.

**Files:**

- Create: `tools/sonic/smoke_sonic_dataloader.py`
- Create: `tools/sonic/smoke_sonic_model.py`

**Verify:**

```bash
PYTHONPATH=. python tools/sonic/smoke_sonic_dataloader.py \
  --config DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml \
  --data-root "$SONIC_DATA_ROOT"
```

Commit message:

```bash
git add tools/sonic/smoke_sonic_dataloader.py tools/sonic/smoke_sonic_model.py
git commit -m "test: add SONIC G1 smoke checks"
```

### Task 6: Offline evaluator

**Objective:** Quantify whether predictions are safe/plausible.

**Files:**

- Create: `examples/Real_G1/eval_files/eval_sonic_g1_offline.py`
- Create: `tools/sonic/evaluate_sonic_predictions.py`

**Verify:**

```bash
PYTHONPATH=. python tools/sonic/evaluate_sonic_predictions.py --help
PYTHONPATH=. python examples/Real_G1/eval_files/eval_sonic_g1_offline.py --help
```

Commit message:

```bash
git add examples/Real_G1/eval_files/eval_sonic_g1_offline.py tools/sonic/evaluate_sonic_predictions.py
git commit -m "eval: add SONIC G1 offline metrics"
```

### Task 7: WebSocket protocol fix/test

**Objective:** Ensure DiT4DiT server can be called by a SONIC-side client deterministically.

**Files:**

- Modify: `deployment/model_server/tools/websocket_policy_server.py`
- Create: `tests/test_websocket_policy_payload.py`

**Verify:**

```bash
PYTHONPATH=. pytest tests/test_websocket_policy_payload.py -q
```

Commit message:

```bash
git add deployment/model_server/tools/websocket_policy_server.py tests/test_websocket_policy_payload.py
git commit -m "fix: accept structured policy server payloads"
```

---

## Final acceptance criteria for this branch

This DiT4DiT branch is complete when:

- `pytest` passes for SONIC schema/config/protocol tests.
- A real or synthetic SONIC LeRobot sample loads with `[40, 78]` action shape.
- `dit4dit_sonic_g1.yaml` loads and points artifacts outside home.
- Tiny overfit demonstrates decreasing loss.
- Offline evaluator produces per-slice metrics and range-violation reports.
- A trained checkpoint can run through the DiT4DiT policy server and return `[1, 40, 78]`.
- The remaining robot-side work is a thin adapter in GR00T-WholeBodyControl, not a science project smuggled into deployment.

## Immediate next action

Start with Task 1 and Task 2. If no SONIC data exists yet, implement schema/tests and the dataset inspector against a tiny synthetic fixture first, then use GR00T-WholeBodyControl to export a minimal real dataset.
