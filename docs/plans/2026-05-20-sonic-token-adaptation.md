# SONIC Token Adaptation Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Adapt DiT4DiT to train and evaluate a policy that predicts SONIC-compatible continuous latent action chunks for Unitree G1-style whole-body control.

**Architecture:** Keep the robot/control stack and the model/training stack separated. This branch in DiT4DiT owns the dataset adapter, action schema, model config, smoke training, and offline replay metrics for SONIC's 78D continuous action contract. GR00T-WholeBodyControl remains the source of truth for deployment and should only receive a small policy-client adapter after this branch proves the interface offline.

**Tech Stack:** DiT4DiT, Cosmos-Predict2.5, LeRobot-style datasets, PyTorch, Accelerate/DeepSpeed, SONIC/GR00T VLA data exported from `NVlabs/GR00T-WholeBodyControl`.

---

## Current branch and source repos

- DiT4DiT worktree: `/home/daniel/dev/dit4dit/sonic-token-actions`
- DiT4DiT branch: `experiment/sonic-token-actions`
- DiT4DiT fork remote: `fork = git@github.com:DanielLSM/DiT4DiT.git`
- Upstream DiT4DiT remote: `origin = https://github.com/Mondo-Robotics/DiT4DiT.git` with push disabled
- SONIC/GR00T reference worktree: `/home/daniel/dev/groot-wholebodycontrol/sonic-vla-pipeline`
- SONIC/GR00T reference branch: `analysis/sonic-vla-pipeline`

## SONIC action contract to preserve

The target output is **continuous**, not categorical token IDs.

Per action step:

- `action.motion_token`: 64 float values in SONIC latent token space
- `action.left_hand_joints`: 7 float values
- `action.right_hand_joints`: 7 float values
- total: 78 float values

Canonical flat layout inside DiT4DiT:

```python
action[..., 0:64] = action.motion_token
action[..., 64:71] = action.left_hand_joints
action[..., 71:78] = action.right_hand_joints
```

Target deployment horizon is 40 steps because the SONIC VLA path consumes a 40-step latent action chunk. The first smoke tests may use a shorter horizon only if needed to validate the loader/model path cheaply, but the plan should converge to `[B, 40, 78]` before any SONIC-side deployment work.

## Design rule

Do **not** vendor GR00T-WholeBodyControl into DiT4DiT and do **not** vendor DiT4DiT into GR00T-WholeBodyControl during this phase. The interface boundary is the dataset/action tensor:

```text
SONIC VLA data export -> DiT4DiT sample -> DiT4DiT predicts [H, 78] -> offline split/range/replay checks
```

Only after offline checks pass should a separate GR00T-WholeBodyControl branch add a thin `dit4dit` policy backend for live inference.

## Phase 0: Pin the interface and avoid false starts

**Objective:** Make the action format explicit before touching model training.

**Files likely touched:**

- Create: `docs/sonic_token_action_contract.md`
- Create: `tests/test_sonic_action_schema.py`
- Create: `DiT4DiT/dataloader/gr00t_lerobot/sonic_action_schema.py`

**High-level tasks:**

1. Add a small `SonicActionSchema` helper with constants:
   - `MOTION_TOKEN_DIM = 64`
   - `LEFT_HAND_DIM = 7`
   - `RIGHT_HAND_DIM = 7`
   - `ACTION_DIM = 78`
   - `TARGET_HORIZON = 40`
2. Add `split_sonic_action(flat_action)` and `concat_sonic_action(parts)` helpers.
3. Unit-test round-trip split/concat for arrays shaped `[78]`, `[H, 78]`, and `[B, H, 78]`.
4. Document that "token" here means **continuous SONIC latent token space**, not a discrete ID.

**Verification:**

```bash
cd /home/daniel/dev/dit4dit/sonic-token-actions
python -m pytest tests/test_sonic_action_schema.py -q
```

Expected: schema tests pass without requiring CUDA, Cosmos, or a dataset.

## Phase 1: Add a SONIC data config to DiT4DiT

**Objective:** Teach DiT4DiT's LeRobot-style dataloader about SONIC VLA keys and dimensions.

**Relevant existing files:**

- `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`
- `DiT4DiT/dataloader/gr00t_lerobot/datasets.py`
- `DiT4DiT/dataloader/gr00t_lerobot/transform/state_action.py`
- `DiT4DiT/dataloader/gr00t_lerobot/transform/concat.py`
- `DiT4DiT/config/real_robot/dit4dit_g1.yaml`
- `examples/Real_G1/eval_files/eval_policy.py`

**High-level tasks:**

1. Add a new `UnitreeG1SonicTokenDataConfig` in `data_config.py`.
2. Use SONIC action keys:
   - `action.motion_token`
   - `action.left_hand_joints`
   - `action.right_hand_joints`
3. Use the state/video/language keys produced by the GR00T-WholeBodyControl VLA export. Do not invent these; confirm them against an exported sample before finalizing the config.
4. Register the config in `ROBOT_TYPE_CONFIG_MAP` as something explicit, e.g. `unitree_g1_sonic_token`.
5. Set `action_indices = list(range(40))` for the final path. If the current loader assumes 16 strongly, add a temporary smoke config and leave a TODO; do not silently truncate the SONIC action contract.

**Verification:**

- A synthetic or tiny exported sample loads through `LeRobotSingleDataset`.
- `action` after transforms has final dimension 78.
- All three SONIC slices survive normalization/unnormalization with stable names.

## Phase 2: Create a SONIC training config

**Objective:** Add a DiT4DiT config that targets 78D SONIC action chunks.

**Files likely touched:**

- Create: `DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml`
- Create: `examples/Real_G1/train_files/run_sonic_token_policy.sh`

**Config changes from `dit4dit_g1.yaml`:**

- `framework.action_model.action_dim: 78`
- `framework.action_model.action_horizon: 40`
- `framework.action_model.future_action_window_size: 39`
- `datasets.vla_data.max_action_dim: 78`
- `datasets.vla_data.data_mix: sonic_token_g1` or equivalent explicit dataset mix name
- `datasets.vla_data.action_type: sonic_latent_token_plus_hands`
- `datasets.vla_data.action_video_freq_ratio`: confirm from exported data before fixing; do not copy the G1 default blindly
- `datasets.vla_data.video_delta_indices`: align with the model's video conditioning and the 40-step action target

**Open design check:** DiT4DiT's current real G1 config uses `action_horizon: 16` and `max_action_dim: 32`. Before full training, verify whether memory and sequence-length settings support `40 x 78` cleanly. If not, make the first milestone a tiny debug config and record the limitation.

## Phase 3: Dataset conversion/export boundary

**Objective:** Keep the GR00T/SONIC repo as the data source, but make DiT4DiT consume the output without knowing robot deployment internals.

**Preferred boundary:** a LeRobot-style dataset directory outside `$HOME`, with explicit feature names matching the SONIC contract.

**Storage rule:** generated datasets, converted shards, checkpoints, logs, W&B artifacts, and caches must not live under `/home/daniel` or any other home directory. Use a scratch/project/data path on the target machine. On Clariden, use the appropriate project/scratch filesystem for the run; do not write large artifacts into the git checkout.

**High-level tasks:**

1. Inspect one real SONIC VLA export from GR00T-WholeBodyControl.
2. Confirm exact feature keys, shapes, dtypes, FPS, horizon, and language key.
3. Write a converter only if the exported dataset is not already compatible with DiT4DiT's LeRobot loader.
4. Add a manifest/sanity script that reports:
   - number of trajectories
   - video keys and shapes
   - state keys and shapes
   - action keys and shapes
   - min/max/mean/std for motion tokens and hands
   - percentage of motion-token values outside the safe deployment bound used by SONIC inference

**Files likely created:**

- `tools/sonic/inspect_sonic_lerobot_dataset.py`
- `tools/sonic/convert_sonic_to_dit4dit.py` only if needed
- `tests/test_sonic_dataset_contract.py`

## Phase 4: CPU/one-batch smoke tests

**Objective:** Prove data and shape flow before GPU training.

**High-level tasks:**

1. Load one batch from the SONIC dataset config.
2. Assert model input construction succeeds.
3. Assert the target action tensor is `[B, 40, 78]` for the final config.
4. If full model import requires GPU/Cosmos weights, add a lightweight dataloader-only smoke test first.

**Verification commands:**

```bash
cd /home/daniel/dev/dit4dit/sonic-token-actions
PYTHONPATH=. python tools/sonic/inspect_sonic_lerobot_dataset.py --dataset-root /path/outside/home/to/sonic_dataset
PYTHONPATH=. python -m pytest tests/test_sonic_action_schema.py tests/test_sonic_dataset_contract.py -q
```

## Phase 5: Tiny overfit run

**Objective:** Confirm DiT4DiT can learn the SONIC target tensor on a tiny subset before wasting cluster allocation.

**Pilot budget:**

- Dataset: 10-50 trajectories, fixed small subset
- GPUs: 1 GPU if feasible; otherwise one short multi-GPU run
- Steps: enough to show training loss decreases clearly, not a full model run
- Artifacts: scratch/project path outside home
- Decision rule: continue only if train loss decreases and predicted action tensors have sane ranges and smoothness

**Verification metrics:**

- total action MSE
- motion-token MSE
- left-hand MSE
- right-hand MSE
- max absolute motion-token value
- percentage of token values outside SONIC's expected safe range
- temporal smoothness, e.g. mean `||a[t+1] - a[t]||`

## Phase 6: Offline replay/evaluation

**Objective:** Evaluate whether predictions are plausible before creating any SONIC-side deployment branch.

**Files likely created:**

- `examples/Real_G1/eval_files/eval_sonic_token_policy.py`
- `tools/sonic/evaluate_sonic_token_predictions.py`

**High-level tasks:**

1. Run the trained checkpoint on held-out logged trajectories.
2. Save predicted `[H, 78]` chunks to a scratch/project output path.
3. Split predictions into motion token and hand fields.
4. Compare against logged ground truth.
5. Generate a short metrics report.

**Decision gate before SONIC repo work:**

Create a GR00T-WholeBodyControl deployment branch only if:

- predicted shape is consistently `[H, 78]`;
- motion-token and hand-joint errors are stable on held-out trajectories;
- range violations are rare and understood;
- temporal smoothness is not obviously pathological;
- the model can run inference at a rate compatible with the downstream SONIC client design.

## Phase 7: Only then add a SONIC-side adapter

This phase should happen in a separate GR00T-WholeBodyControl branch, not here, and only after Phase 6 passes.

Likely branch name:

```text
experiment/dit4dit-policy-client
```

SONIC-side scope should be intentionally small:

- load/call a trained DiT4DiT checkpoint or policy server;
- convert `[H, 78]` to:
  - `action.motion_token`
  - `action.left_hand_joints`
  - `action.right_hand_joints`
- reuse existing SONIC publish/range-check logic;
- add mock/sim tests before any robot run.

## Non-goals for this branch

- Do not modify SONIC C++ deployment.
- Do not change SONIC's latent token representation.
- Do not add a discrete/categorical token head unless a separate experiment explicitly targets FSQ indices.
- Do not vendor GR00T-WholeBodyControl into DiT4DiT.
- Do not start full Clariden training before data-shape and overfit checks pass.

## First implementation milestone

The first useful PR/commit sequence should be:

1. Add `sonic_action_schema.py` and schema tests.
2. Add `UnitreeG1SonicTokenDataConfig` registered as `unitree_g1_sonic_token`.
3. Add `dit4dit_sonic_g1.yaml` with 78D/40-step target settings.
4. Add a dataloader smoke script that prints one-batch shapes.
5. Add a tiny-overfit run script with all outputs directed outside `$HOME`.

That milestone is deliberately boring. Boring interfaces keep robots alive.
