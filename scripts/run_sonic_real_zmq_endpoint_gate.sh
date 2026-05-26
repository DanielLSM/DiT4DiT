#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/iopsstor/scratch/cscs/dsimoes/dit4dit/data/sonic-g1-smoke}"
DATA_MIX="${DATA_MIX:-sonic_g1_smoke}"
OUT_DIR="${OUT_DIR:-/iopsstor/scratch/cscs/dsimoes/dit4dit/runs/smoke/sonic-real-zmq-endpoint-$(date +%Y%m%d_%H%M%S)}"
SONIC_REPO="${SONIC_REPO:-/iopsstor/scratch/cscs/dsimoes/dit4dit/sonic-dit4dit}"
PORT="${PORT:-6901}"
EXPECTED_UPDATES="${EXPECTED_UPDATES:-40}"
PUBLISH_RATE_HZ="${PUBLISH_RATE_HZ:-10}"
PUBLISH_WARMUP_SEC="${PUBLISH_WARMUP_SEC:-1.5}"
POLICY_STEPS="${POLICY_STEPS:-3}"
CHECKPOINT="${CHECKPOINT:-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"

mkdir -p "$OUT_DIR"

HARNESS_SRC="$SONIC_REPO/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/zmq_endpoint_v4_real_interface_test.cpp"
SONIC_INCLUDE="$SONIC_REPO/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include"
HARNESS_BIN="$OUT_DIR/zmq_endpoint_v4_real_interface_test"
HARNESS_JSON="$OUT_DIR/real_zmq_endpoint.json"
HARNESS_LOG="$OUT_DIR/real_zmq_endpoint.log"
POLICY_JSON="$OUT_DIR/dit4dit_policy.json"
POLICY_NPZ="$OUT_DIR/dit4dit_policy.npz"
POLICY_VIDEO="$OUT_DIR/dit4dit_policy.mp4"
SUMMARY_JSON="$OUT_DIR/summary.json"

if [[ ! -f "$HARNESS_SRC" ]]; then
  echo "Missing SONIC real-interface harness: $HARNESS_SRC" >&2
  exit 10
fi

printf '[gate] compiler: '
command -v g++
printf '[gate] python: '
command -v python
printf '[gate] SONIC repo: %s\n' "$SONIC_REPO"
printf '[gate] DiT4DiT data root: %s\n' "$DATA_ROOT"
printf '[gate] data mix: %s\n' "$DATA_MIX"
if [[ -n "$CHECKPOINT" ]]; then
  printf '[gate] action-head checkpoint: %s\n' "$CHECKPOINT"
fi
printf '[gate] output dir: %s\n' "$OUT_DIR"

VENDOR_CPP_HEADERS="${VENDOR_CPP_HEADERS:-/iopsstor/scratch/cscs/dsimoes/dit4dit/vendor/cpp-headers/include}"
CXX_INCLUDES=("-I$SONIC_INCLUDE")
if [[ -d "$VENDOR_CPP_HEADERS" ]]; then
  CXX_INCLUDES=("-I$VENDOR_CPP_HEADERS" "${CXX_INCLUDES[@]}")
  printf '[gate] using vendor C++ headers: %s\n' "$VENDOR_CPP_HEADERS"
fi

ZMQ_LINK_ARG="${ZMQ_LINK_ARG:--l:libzmq.so.5}"
printf '[gate] ZMQ link arg: %s\n' "$ZMQ_LINK_ARG"

g++ -std=c++20 -O2 -pthread \
  "${CXX_INCLUDES[@]}" \
  "$HARNESS_SRC" \
  "$ZMQ_LINK_ARG" \
  -o "$HARNESS_BIN"

"$HARNESS_BIN" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --topic pose \
  --expected-updates "$EXPECTED_UPDATES" \
  --timeout-sec 30 \
  --poll-hz 200 \
  --output-json "$HARNESS_JSON" \
  >"$HARNESS_LOG" 2>&1 &
HARNESS_PID=$!

cleanup() {
  if kill -0 "$HARNESS_PID" 2>/dev/null; then
    kill "$HARNESS_PID" 2>/dev/null || true
    wait "$HARNESS_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Give ZMQEndpointInterface time to connect before the PUB socket starts sending.
sleep 1

POLICY_ARGS=(
  --data-root "$DATA_ROOT"
  --data-mix "$DATA_MIX"
  --steps "$POLICY_STEPS"
  --batch-size "$BATCH_SIZE"
  --video-backend "$VIDEO_BACKEND"
  --output-json "$POLICY_JSON"
  --output-npz "$POLICY_NPZ"
  --output-video "$POLICY_VIDEO"
  --publish-zmq-host '*'
  --publish-zmq-port "$PORT"
  --publish-rate-hz "$PUBLISH_RATE_HZ"
  --publish-warmup-sec "$PUBLISH_WARMUP_SEC"
)
if [[ -n "$CHECKPOINT" ]]; then
  POLICY_ARGS+=(--checkpoint "$CHECKPOINT")
fi

python scripts/smoke_sonic_policy_to_zmq.py "${POLICY_ARGS[@]}"

set +e
wait "$HARNESS_PID"
HARNESS_STATUS=$?
set -e
trap - EXIT

python - <<PY
import json
from pathlib import Path
out = Path("$OUT_DIR")
summary = {
    "ok": False,
    "gate": "dit4dit_policy_to_real_sonic_zmq_endpoint",
    "out_dir": str(out),
    "policy_json": "$POLICY_JSON",
    "policy_npz": "$POLICY_NPZ",
    "policy_video": "$POLICY_VIDEO",
    "real_endpoint_json": "$HARNESS_JSON",
    "real_endpoint_log": "$HARNESS_LOG",
    "harness_status": $HARNESS_STATUS,
}
for key, path in [("policy", Path("$POLICY_JSON")), ("real_endpoint", Path("$HARNESS_JSON"))]:
    if path.exists():
        summary[key] = json.loads(path.read_text())
    else:
        summary[key] = {"ok": False, "error": f"missing {path}"}
summary["ok"] = bool(summary.get("policy", {}).get("ok")) and bool(summary.get("real_endpoint", {}).get("ok")) and summary["harness_status"] == 0
Path("$SUMMARY_JSON").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [[ "$HARNESS_STATUS" -ne 0 ]]; then
  echo "[gate] harness failed with status $HARNESS_STATUS; log follows:" >&2
  sed -n '1,240p' "$HARNESS_LOG" >&2 || true
  exit "$HARNESS_STATUS"
fi

echo "[gate] summary: $SUMMARY_JSON"
