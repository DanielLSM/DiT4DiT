# DiT4DiT SONIC G1 Execution Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Train and evaluate a DiT4DiT policy that predicts SONIC's 78D continuous latent-action chunks for the Unitree G1, then run it in-process inside SONIC's existing inference loop — *only* after the offline interface is proven.

**Architecture:** Two repos, one tensor boundary, in-process integration.

- `DiT4DiT` (this repo, `experiment/sonic-token-actions`) owns: action schema, data config, training config, smoke/overfit, offline replay metrics, and a checkpoint + `stats.json` artifact.
- `GR00T-WholeBodyControl` owns: data export, deployment, safety, publishing. Will receive a small in-process adapter in a later branch (`experiment/dit4dit-policy-client`) that imports DiT4DiT directly — no network protocol between them.
- Interface: a LeRobot v2.1 dataset directory for training and a `(checkpoint, stats.json)` pair for inference. Nothing is vendored across.

**Integration rationale:** SONIC's existing VLA loop ([run_vla_inference.py](file:///home/daniel/dev/groot-wholebodycontrol/sonic-vla-pipeline/gear_sonic/scripts/run_vla_inference.py)) talks to an Isaac-GR00T `PolicyClient` over ZMQ. For the first end-to-end bring-up we fork that script and replace the `PolicyClient.get_action(...)` call with an in-process `vla.predict_action(...)`. No new protocol, no server process, no payload schema bug to chase. A separate-machine network deployment (Isaac-GR00T-compatible ZMQ server) is a later concern, not part of this plan.

**Tech stack:** DiT4DiT, Cosmos-Predict2.5-2B, PyTorch 2.7 / CUDA 12.8, Accelerate/DeepSpeed, SONIC VLA data exported from `NVlabs/GR00T-WholeBodyControl`.

---

## SONIC action contract (frozen)

Per action step, 78 continuous floats:

```python
action[..., 0:64]   = action.motion_token        # SONIC latent token (FSQ-style)
action[..., 64:71]  = action.left_hand_joints    # 7 continuous joint targets
action[..., 71:78]  = action.right_hand_joints   # 7 continuous joint targets
```

Target deployment horizon: `H = 40`. Final model output shape: `[B, 40, 78]`. A shorter `H` is allowed only for smoke tests and must never reach the robot.

**Continuous, not categorical.** DiT4DiT's flow-matching head consumes `example["action"]` as a continuous tensor and returns `normalized_actions` ([DiT4DiT/model/framework/DiT4DiT.py:176](../../DiT4DiT/model/framework/DiT4DiT.py#L176)). The 64-D motion latent stays a continuous vector end-to-end; it is not serialized through any discrete LLM vocabulary.

---

## Evidence already verified in this repo

- [DiT4DiT/config/real_robot/dit4dit_g1.yaml](../../DiT4DiT/config/real_robot/dit4dit_g1.yaml): currently `action_dim:32, action_horizon:16, future_action_window_size:15, max_action_dim:32, data_mix:real_robot_all, action_video_freq_ratio:2, video_delta_indices:[0..16]`. Not SONIC-compatible.
- [DiT4DiT/dataloader/gr00t_lerobot/data_config.py:1029](../../DiT4DiT/dataloader/gr00t_lerobot/data_config.py#L1029): `UnitreeG1AlohaOnlyArmsDataConfig` exists, uses `video.ego_view`, `annotation.human.task_description`, `action_indices = list(range(16))`, `min_max` normalization for all action keys.
- [DiT4DiT/dataloader/gr00t_lerobot/mixtures.py:391](../../DiT4DiT/dataloader/gr00t_lerobot/mixtures.py#L391): `real_robot_all` lists 7 folder names (pnp_eggplant_lh_200ep_26_1_28, pnp_corn_middle_drawer, …) that are *not* publicly downloadable from this repo and target ALOHA arms, not SONIC.
- `/home/daniel/dev/groot-wholebodycontrol/sonic-vla-pipeline/gear_sonic/data/features_sonic_vla.py`: confirms the SONIC export writes exactly `action.motion_token` (64), `action.left_hand_joints` (7), `action.right_hand_joints` (7), `annotation.human.task_description`, at `FPS=50`.
- `/home/daniel/dev/groot-wholebodycontrol/sonic-vla-pipeline/gear_sonic/scripts/run_vla_inference.py`: existing inference loop is ZMQ end-to-end and uses Isaac-GR00T's `PolicyClient` over REQ/REP. This is what the in-process adapter will fork.

**Data conclusion:** training data must come from the GR00T-WholeBodyControl exporter. The existing `real_robot_all` mixture is evidence the repo loads real G1 data, but provides neither the data nor the right action schema.

---

## Storage rule (applies to every phase)

Datasets, checkpoints, run logs, W&B artifacts, and caches must not live under `/home/daniel` or inside the repo. Set these once per shell:

```bash
export SONIC_DATA_ROOT=/cluster/scratch/$USER/dit4dit-data/sonic-g1-lerobot
export DIT4DIT_RUN_ROOT=/cluster/scratch/$USER/dit4dit-runs/sonic-g1
export COSMOS_ROOT=/cluster/scratch/$USER/models/Cosmos-Predict2.5-2B
```

On non-Clariden machines, substitute the appropriate scratch/project path. `results/Checkpoints` is *never* an acceptable target.

---

## Phase 0: Empirical pre-flight — kill four unknowns before writing config

**Why this phase exists:** four config values cannot be reasoned from docs alone. Pinning them from one inspected episode is ~30 min of work and prevents an entire week of "why doesn't the loader return what I expect."

**Files:**

- Create: `tools/sonic/inspect_sonic_lerobot_dataset.py`
- Create: `docs/sonic_g1_data_inventory.md`

**The four unknowns to resolve and record in the inventory:**

1. **Video FPS and `action_video_freq_ratio`.** SONIC export is `FPS = 50` per `features_sonic_vla.py`. DiT4DiT's G1 default uses `action_video_freq_ratio: 2` and `video_delta_indices: [0..16]`. The right value depends on what `meta/info.json` in the exported dataset actually reports for video FPS vs action FPS. Record the observed ratio.
2. **Motion-token distribution.** Plot per-dim histograms over ≥3 episodes. If distribution is roughly unimodal/bounded, `min_max` is fine. If it's bimodal, heavy-tailed, or quantized (likely for FSQ latents), use `q99` or document a custom mode. **Do not pick a normalizer before looking at the data.**
3. **Real state dimension.** Plan B guessed `state_dim: 64`. The actual SONIC state comes from joint groups `left_leg + right_leg + waist + left_arm + left_hand + right_arm + right_hand` plus wrist pose / root orientation. Read `meta/modality.json` from one export and record the *true* sum.
4. **Exact key names.** Confirm video key (`video.ego_view` vs `video.image` vs other), confirm `annotation.human.task_description` is present, confirm there are no surprises like `action.wbc` polluting the action group (it's referenced in features but is *not* part of the 78D contract).

**Inspector script requirements** (`tools/sonic/inspect_sonic_lerobot_dataset.py`):

- Args: `--dataset-root`, `--limit-episodes`, `--save-histograms`.
- Prints: dataset path, LeRobot version, episode count, video FPS, action FPS, video keys + shapes, state keys + shapes, language key, action keys + dims, min/max/mean/std per action key.
- Writes: `docs/sonic_g1_data_inventory.md` with all of the above, plus a markdown table per-dim min/max/p1/p99 for `action.motion_token`.
- Asserts: every episode has ≥40 future action steps after its observation window.

**Data acquisition: three tiers, pick the cheapest one that unblocks the next phase.**

There is no public SONIC VLA dataset (HuggingFace has the SONIC controller checkpoint and BONES-SEED motions, but no LeRobot dataset with `action.motion_token + hand_joints`). So data has to come from one of:

1. **Synthetic fixture (default for Phases 1–6).** Generate a tiny LeRobot v2.1 dataset with the right keys, shapes, dtypes, and metadata — but random-noise content. Sufficient for: dataloader shape contract, model forward smoke, and the tiny-overfit gate (the model *can* overfit random data with the right shape; the loss-down signal proves the head and config are wired correctly). Insufficient for: anything in Phase 7 (offline eval).

   Create `tools/sonic/make_synthetic_sonic_dataset.py` that writes a LeRobot v2.1 directory under `$SONIC_DATA_ROOT/synthetic-tiny/` with:
   - 20 episodes × 60 steps each
   - `observation.images.ego_view`: random `uint8` `[H, W, 3]` at the configured FPS
   - `action.motion_token`: random `float32 [T, 64]` from a bounded distribution (e.g. `tanh(N(0, 1))`)
   - `action.left_hand_joints` / `action.right_hand_joints`: random `float32 [T, 7]` in `[0, 1]`
   - `annotation.human.task_description`: one of three canned strings
   - `meta/info.json`, `meta/modality.json`, `meta/episodes.jsonl`, `meta/stats.json` populated correctly so DiT4DiT's loader accepts it

2. **MuJoCo sim collection.** If you have the SONIC sim2sim setup working, `gear_sonic/scripts/run_sim_loop.py` + `run_data_exporter.py` can collect real-shape data without a physical robot, using scripted or teleoperated actions in MuJoCo. Better than synthetic for Phase 7 trial runs; still not a substitute for real teleop data on real hardware. The docs ([data_collection.md](file:///home/daniel/dev/groot-wholebodycontrol/main/docs/source/tutorials/data_collection.md)) confirm sim mode publishes camera frames automatically — no camera server needed.

3. **Real teleop collection.** Required to *trust* Phase 7 metrics for a real-robot decision. Workstation + PICO VR + camera server on a Jetson + a SONIC-controlled G1. This is the heavyweight option — set it up only after the synthetic-fixture path has proven the model architecture and training loop are sound.

**Commands:**

```bash
# Tier 1: synthetic (use this first)
PYTHONPATH=. python tools/sonic/make_synthetic_sonic_dataset.py \
  --output-dir "$SONIC_DATA_ROOT/synthetic-tiny" \
  --num-episodes 20 --steps-per-episode 60

# Tier 2: sim (only if sim2sim already works on your box)
cd /home/daniel/dev/groot-wholebodycontrol/main
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "pick up the cup" \
  --dataset-name sonic_g1_pick_cup_sim \
  --root-output-dir "$SONIC_DATA_ROOT"

# Tier 3: real teleop (only when ready to commit to bring-up)
# See docs/source/tutorials/data_collection.md in the GR00T-WholeBodyControl repo.
```

`launch_data_collection.py` currently exposes only `dataset_name` in its dataclass — not `root_output_dir`. For serious collection (tier 2 or 3), call `run_data_exporter.py` directly with `--root-output-dir`, or patch the launcher to pass it through.

**Verification (against whichever tier you ran):**

```bash
PYTHONPATH=. python tools/sonic/inspect_sonic_lerobot_dataset.py \
  --dataset-root "$SONIC_DATA_ROOT/synthetic-tiny" \
  --limit-episodes 3 --save-histograms
```

Exit 0 and `docs/sonic_g1_data_inventory.md` populated. For synthetic data, the inventory will record arbitrary stats — that's fine for Phases 1–6; the inventory must be **regenerated against real data before Phase 7**.

---

## Phase 1: Action schema helper

**Objective:** Make the 78D contract a testable Python module.

**Files:**

- Create: `DiT4DiT/dataloader/gr00t_lerobot/sonic_action_schema.py`
- Create: `tests/test_sonic_action_schema.py`

**Module contents:**

```python
MOTION_TOKEN_DIM = 64
LEFT_HAND_DIM = 7
RIGHT_HAND_DIM = 7
ACTION_DIM = MOTION_TOKEN_DIM + LEFT_HAND_DIM + RIGHT_HAND_DIM  # 78
TARGET_HORIZON = 40

MOTION_SLICE = slice(0, 64)
LEFT_HAND_SLICE = slice(64, 71)
RIGHT_HAND_SLICE = slice(71, 78)

def split_sonic_action(action):
    """Accept [..., 78], return dict with motion_token, left_hand_joints, right_hand_joints."""

def concat_sonic_action(motion_token, left_hand_joints, right_hand_joints):
    """Return [..., 78] in canonical SONIC ordering."""
```

**Tests:**

- Round-trip split→concat for shapes `[78]`, `[40, 78]`, `[B, 40, 78]`.
- Wrong last dim raises a clear assertion.
- Slice constants sum to 78.

**Verify:** `PYTHONPATH=. pytest tests/test_sonic_action_schema.py -q` — passes with no CUDA, no Cosmos, no dataset.

---

## Phase 2: SONIC data config and mixture

**Objective:** Teach DiT4DiT's LeRobot loader the SONIC keys, horizon, and normalization.

**Files:**

- Modify: `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`
- Modify: `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`
- Create: `tests/test_sonic_data_config.py`

**Add `UnitreeG1SonicTokenDataConfig`** (mirror `UnitreeG1AlohaOnlyArmsDataConfig`, replace action keys, set 40-step horizon):

```python
class UnitreeG1SonicTokenDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]  # confirm from Phase 0
    state_keys = [
        # exact list from meta/modality.json — read, don't invent
    ]
    action_keys = [
        "action.motion_token",
        "action.left_hand_joints",
        "action.right_hand_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(40))  # TARGET_HORIZON
```

**Normalization** (modes chosen from Phase 0 histograms, not assumed):

```python
StateActionTransform(
    apply_to=self.action_keys,
    normalization_modes={
        "action.motion_token":     <"min_max" or "q99" — from inventory>,
        "action.left_hand_joints":  "min_max",  # bounded joint range
        "action.right_hand_joints": "min_max",
    },
)
```

Do *not* use `binary` for hand joints; do *not* default to `min_max` for motion tokens without looking at the histograms — FSQ latents can have outliers that distort min/max.

**Register:**

```python
ROBOT_TYPE_CONFIG_MAP["unitree_g1_sonic_token"] = UnitreeG1SonicTokenDataConfig()
```

**Mixture** in `mixtures.py`:

```python
"sonic_g1_token_all": [
    ("<folder-from-inventory>", 1.0, "unitree_g1_sonic_token"),
]
```

List every folder explicitly. No globbing — missing data should be loud.

**Embodiment tag:** use `NEW_EMBODIMENT`. The action semantics are new; do not reuse the existing G1 arm-joint embedding.

**Tests** (`tests/test_sonic_data_config.py`):

- `ROBOT_TYPE_CONFIG_MAP["unitree_g1_sonic_token"]` exists and exposes the right keys.
- `DATASET_NAMED_MIXTURES["sonic_g1_token_all"]` lists at least one folder, all tagged `unitree_g1_sonic_token`.
- `len(action_indices) == 40`.

**Verify:**

```bash
PYTHONPATH=. pytest tests/test_sonic_data_config.py -q
PYTHONPATH=. python tools/sonic/inspect_sonic_lerobot_dataset.py \
  --dataset-root "$SONIC_DATA_ROOT" --robot-type unitree_g1_sonic_token
```

---

## Phase 3: SONIC training config

**Objective:** A YAML config that trains DiT4DiT on 78D / 40-step actions.

**Files:**

- Create: `DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml`
- Create: `examples/Real_G1/train_files/run_sonic_g1.sh`
- Create: `examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh`

**Required diffs from `dit4dit_g1.yaml`:**

```yaml
framework:
  action_model:
    action_dim: 78
    state_dim: <true sum from Phase 0, not 64>
    future_action_window_size: 39
    action_horizon: 40

datasets:
  vla_data:
    data_root_dir: ${oc.env:SONIC_DATA_ROOT}
    data_mix: sonic_g1_token_all
    action_type: sonic_latent_token_plus_hands
    max_action_dim: 78
    action_video_freq_ratio: <observed from Phase 0>
    video_delta_indices: <re-derived for 40-step horizon and observed FPS>
```

**Tiny-overfit trainer settings:**

```yaml
trainer:
  max_train_steps: 500
  save_interval: 250
  eval_interval: 50
  logging_frequency: 10
  freeze_modules: "backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"
```

**Run scripts:** must export `SONIC_DATA_ROOT`, `DIT4DIT_RUN_ROOT`, `COSMOS_ROOT` and pass `run_root_dir=$DIT4DIT_RUN_ROOT/<run-id>`. The default `results/Checkpoints` is overridden, period.

**Verify:**

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
bash -n examples/Real_G1/train_files/run_sonic_g1.sh
bash -n examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh
```

---

## Phase 4: Dataloader smoke

**Objective:** Prove a single batch comes back as `[B, 40, 78]` before importing the model.

**Files:**

- Create: `tools/sonic/smoke_sonic_dataloader.py`
- Create: `tests/test_sonic_dataset_contract.py`

The smoke script must:

- Load `dit4dit_sonic_g1.yaml`.
- Build the dataset via `get_vla_dataset` and a DataLoader with `num_workers=0`.
- Pull one sample and one batch.
- Print keys, shapes, dtypes.
- Assert `example["action"].shape[-1] == 78`, `example["action"].shape[-2] == 40`.
- Use `split_sonic_action` to print per-slice min/max — sanity check normalization didn't collapse the motion-token slice.

**Verify:**

```bash
PYTHONPATH=. python tools/sonic/smoke_sonic_dataloader.py \
  --config DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml \
  --data-root "$SONIC_DATA_ROOT" --num-workers 0
```

Expected lines in output:

```
action shape: [40, 78]
motion_token shape: [40, 64]
left_hand_joints shape: [40, 7]
right_hand_joints shape: [40, 7]
```

---

## Phase 5: Model forward / predict smoke

**Objective:** Confirm DiT4DiT can construct and sample `[1, 40, 78]` on the SONIC config.

**Files:**

- Create: `tools/sonic/smoke_sonic_model.py`

The script must:

- Load the SONIC config.
- Build the model from `$COSMOS_ROOT`.
- Run one batch forward → `action_loss` is finite.
- Call `predict_action([example])` → `normalized_actions` has shape `[1, 40, 78]`.

**Verify:**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/sonic/smoke_sonic_model.py \
  --config DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml \
  --cosmos-root "$COSMOS_ROOT" --data-root "$SONIC_DATA_ROOT"
```

---

## Phase 6: Tiny overfit

**Objective:** Confirm the head can fit a small SONIC-shaped subset before booking real cluster time.

**Data tier:** synthetic fixture from Phase 0 is sufficient. The pass gate here is *structural* — does the model architecture + training loop converge on a fixed small target? — not predictive of real-robot behavior. Re-running this on real data later is cheap and worth doing once available, but is not the gate for moving to Phase 7.

**Files:**

- Create: `examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh`

**Budget:**

- 10–50 trajectories from `sonic_g1_token_all` (point the mixture at `synthetic-tiny/` for the first pass).
- 1 GPU if it fits; otherwise the smallest multi-GPU run that does.
- 500–2,000 steps, frozen text encoder + VAE.
- Outputs at `$DIT4DIT_RUN_ROOT/tiny-overfit-<date>`.

**Metrics logged each step:**

- Total action MSE; per-slice MSE for motion-token / left hand / right hand.
- Max `|motion_token|`; fraction of motion-token predictions outside SONIC's safe latent range (from Phase 0 inventory).
- Temporal smoothness: mean `||a[t+1] - a[t]||` over the chunk.

**Pass gate:**

- Train loss drops substantially on the tiny subset.
- Per-slice MSE decreases.
- Motion-token range stays bounded; no divergence.
- Predicted chunks are temporally smooth enough to replay.

---

## Phase 7: Held-out offline evaluation + **normalization-stats export contract**

**Objective:** Decide whether the model is worth wiring to a robot. **Also: produce the artifact the SONIC adapter will depend on — the dataset normalization stats.**

**Data tier:** real teleop data (tier 3) or, at minimum, MuJoCo sim collection (tier 2). The synthetic fixture is *not acceptable* here — held-out metrics on random data are meaningless and the `stats.json` produced from synthetic data would mislead the SONIC adapter at deployment. Regenerate `docs/sonic_g1_data_inventory.md` against real data before entering this phase.

**Files:**

- Create: `examples/Real_G1/eval_files/eval_sonic_g1_offline.py`
- Create: `tools/sonic/evaluate_sonic_predictions.py`
- Create: `docs/reports/sonic_g1_offline_eval_template.md`

**Critical: the normalization-stats artifact.**

`predict_action` returns *normalized* actions ([DiT4DiT/model/framework/DiT4DiT.py:225](../../DiT4DiT/model/framework/DiT4DiT.py#L225)). The SONIC adapter must unnormalize using the exact stats from training, or motion tokens land outside the decoder's expected range and the robot moves weirdly. The eval pipeline must therefore:

1. Save `stats.json` next to every checkpoint, with per-action-key min/max (or q99 bounds) used at training time.
2. The format must be readable from outside DiT4DiT — plain JSON, no torch pickles. Example:
   ```json
   {
     "action.motion_token":    {"mode": "q99", "p1": [...64...], "p99": [...64...]},
     "action.left_hand_joints":  {"mode": "min_max", "min": [...7...], "max": [...7...]},
     "action.right_hand_joints": {"mode": "min_max", "min": [...7...], "max": [...7...]}
   }
   ```
3. The SONIC-side adapter loads this file. There is no other contract.

**Offline-eval run:**

```bash
PYTHONPATH=. python examples/Real_G1/eval_files/eval_sonic_g1_offline.py \
  --model_path "$DIT4DIT_RUN_ROOT/<ckpt-dir>/pytorch_model.pt" \
  --stats_path "$DIT4DIT_RUN_ROOT/<ckpt-dir>/stats.json" \
  --dataset_root "$SONIC_DATA_ROOT" \
  --data_config unitree_g1_sonic_token \
  --data_mix sonic_g1_token_all \
  --action_horizon 40 \
  --save_dir "$DIT4DIT_RUN_ROOT/offline-eval/<run-id>"
```

**Report fields** (template `sonic_g1_offline_eval_template.md`):

- Checkpoint path, dataset root, held-out episodes.
- Per-slice MSE (raw + normalized).
- Range-violation rate per motion-token dim.
- Smoothness metrics.
- Example trajectory plots (predicted vs ground truth).
- Inference latency per chunk.
- `stats.json` SHA.

**Decision gate before any robot work:** proceed only if held-out per-slice MSE is stable, range violations are rare and understood, and predicted chunks replay cleanly through the SONIC decoder (if a decoder-only test path exists).

---

## Phase 8: In-process SONIC adapter (separate repo, separate branch)

**Repo:** `/home/daniel/dev/groot-wholebodycontrol/`
**Worktree:** `/home/daniel/dev/groot-wholebodycontrol/dit4dit-policy-client`
**Branch:** `experiment/dit4dit-policy-client`

Do *not* implement this phase in DiT4DiT. The artifacts crossing the line are the trained checkpoint and `stats.json`.

**Approach:** fork `gear_sonic/scripts/run_vla_inference.py` into `run_dit4dit_inference.py`. The fork keeps every ZMQ socket (state SUB, action PUB, camera, keyboard, telemetry) and every existing helper (`prepare_observation_for_eval`, `concat_action`, `should_trigger_new_inference`, `calculate_latency_compensated_index`). Only the policy call changes.

**What to replace:**

- Remove the Isaac-GR00T `PolicyClient` connection (`host`, `port` CLI args, ZMQ REQ socket setup).
- At startup, load DiT4DiT in-process:
  ```python
  from DiT4DiT.model.framework.base_framework import baseframework
  vla = baseframework.from_pretrained(args.ckpt_path).to("cuda").eval()
  stats = json.load(open(args.stats_path))
  ```
- In the inference tick, replace `policy_client.get_action(obs)` with:
  ```python
  out = vla.predict_action(examples=[build_example(obs, prompt)])
  normalized = out["normalized_actions"]              # [1, 40, 78]
  action = unnormalize(normalized, stats)              # apply per-key inverse transform
  motion_token       = action[0, :, 0:64]
  left_hand_joints   = action[0, :, 64:71]
  right_hand_joints  = action[0, :, 71:78]
  ```
- Hand the resulting chunk to the existing `concat_action` / publish path — do not rewrite range checks, latency compensation, or publishing.

**New CLI args** (replacing `host`/`port`):

- `--ckpt-path`: path to the DiT4DiT checkpoint.
- `--stats-path`: path to `stats.json` from Phase 7.
- `--enable-publish`: default `False`. Required to actually push actions to the C++ controller.

**Why no Isaac-GR00T-compatible ZMQ server:** keeping it in-process means one process to launch, no protocol matching, and the SONIC repo only adds one new script plus a small `dit4dit_runner.py` utility. If a future deployment needs the model on a separate box, wrap `vla.predict_action` in an Isaac-GR00T-compatible PolicyServer then; do not pay that cost now.

**Cost of in-process to keep in mind:** if DiT4DiT crashes the inference process exits with it. Acceptable for bring-up and offline replay; revisit if sustained robot work depends on it.

---

## Phase 9: Robot ladder

Only after Phases 0–8 pass.

1. Unit tests: schema split/concat.
2. Dataset test: one real sample emits `[40, 78]`.
3. Model test: `predict_action` returns `[1, 40, 78]`.
4. Adapter smoke: `run_dit4dit_inference.py` runs against logged observations with publish disabled.
5. Offline replay: predictions vs logged SONIC ground truth.
6. Decoder-only test: predicted chunks through the SONIC latent decoder, motors off.
7. Robot dry-run: live perception, inference only, no publish.
8. Shadow mode: publish to log while native controller drives.
9. Low-risk task: reduced speed / range, hands disabled if useful.
10. Full task attempt with operator and e-stop.

**Robot-run checklist (every run):**

- E-stop tested.
- `--enable-publish` is the only path that moves motors.
- Max `|motion_token|` guard active, set from training distribution.
- Hand-joint software limits active.
- Stale-action timeout active.
- Logs land outside `$HOME`.
- Checkpoint SHA, `stats.json` SHA, dataset SHA all recorded with the run.

---

## Non-goals (re-stated)

- No SONIC C++ deployment changes.
- No changes to SONIC's latent-token representation.
- No discrete/categorical action head — DiT4DiT stays continuous.
- No vendoring across repos.
- No full Clariden training before the offline gate passes.

---

## Acceptance criteria for this branch

- `pytest` passes for schema / data-config tests.
- A real SONIC LeRobot sample loads with shape `[40, 78]` through `unitree_g1_sonic_token`.
- `dit4dit_sonic_g1.yaml` parses with the asserted values; all artifact paths resolve outside `$HOME`.
- Tiny overfit shows decreasing per-slice loss on a small subset.
- Offline evaluator produces per-slice metrics, range-violation rate, and a `stats.json` next to the checkpoint.
- In-process call to `vla.predict_action(...)` returns `[1, 40, 78]` and unnormalizes cleanly with the exported `stats.json`.
- The robot-side work is genuinely a forked `run_dit4dit_inference.py` in the other repo, not a science project.

---

## Task-by-task commit sequence

Each task is one PR-sized commit. Verify command must pass before commit.

### Task 1a — Synthetic-fixture generator

- New: `tools/sonic/make_synthetic_sonic_dataset.py`
- Verify: `PYTHONPATH=. python tools/sonic/make_synthetic_sonic_dataset.py --output-dir "$SONIC_DATA_ROOT/synthetic-tiny" --num-episodes 20 --steps-per-episode 60` produces a loadable LeRobot v2.1 directory.
- Commit: `feat: synthetic SONIC fixture for shape/overfit tests`

### Task 1b — Data inventory + inspector

- New: `tools/sonic/inspect_sonic_lerobot_dataset.py`, `docs/sonic_g1_data_inventory.md`
- Verify: `PYTHONPATH=. python tools/sonic/inspect_sonic_lerobot_dataset.py --dataset-root "$SONIC_DATA_ROOT/synthetic-tiny" --limit-episodes 3`
- Commit: `docs: inventory SONIC G1 data contract`

### Task 2 — Action schema

- New: `DiT4DiT/dataloader/gr00t_lerobot/sonic_action_schema.py`, `tests/test_sonic_action_schema.py`
- Verify: `PYTHONPATH=. pytest tests/test_sonic_action_schema.py -q`
- Commit: `feat: add SONIC action schema helpers`

### Task 3 — Data config + mixture

- Modify: `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`, `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`
- New: `tests/test_sonic_data_config.py`
- Verify: `PYTHONPATH=. pytest tests/test_sonic_data_config.py -q`
- Commit: `feat: register SONIC G1 data config`

### Task 4 — Training config + run scripts

- New: `DiT4DiT/config/real_robot/dit4dit_sonic_g1.yaml`, `examples/Real_G1/train_files/run_sonic_g1*.sh`
- Verify: OmegaConf assertion block from Phase 3 + `bash -n`
- Commit: `feat: add SONIC G1 training config`

### Task 5 — Dataloader smoke

- New: `tools/sonic/smoke_sonic_dataloader.py`, `tests/test_sonic_dataset_contract.py`
- Verify: smoke script prints `action shape: [40, 78]`
- Commit: `test: SONIC G1 dataloader contract`

### Task 6 — Model smoke

- New: `tools/sonic/smoke_sonic_model.py`
- Verify: GPU run returns `[1, 40, 78]`
- Commit: `test: SONIC G1 model forward/predict smoke`

### Task 7 — Tiny overfit script

- New: `examples/Real_G1/train_files/run_sonic_g1_tiny_overfit.sh`
- Verify: scripted run produces decreasing per-slice loss on a 10-traj subset
- Commit: `train: SONIC G1 tiny-overfit recipe`

### Task 8 — Offline eval + stats export

- New: `examples/Real_G1/eval_files/eval_sonic_g1_offline.py`, `tools/sonic/evaluate_sonic_predictions.py`, `docs/reports/sonic_g1_offline_eval_template.md`
- Verify: eval run emits per-slice metrics + `stats.json`
- Commit: `eval: SONIC G1 offline metrics + stats export`

---

## Immediate next action

Start at Task 1 (Phase 0). Do not skip the empirical pre-flight — every value it produces feeds Tasks 3 and 4. If no real SONIC export exists yet, run `run_data_exporter.py` once into `$SONIC_DATA_ROOT` before opening the inspector.

Boring interfaces keep robots alive.
