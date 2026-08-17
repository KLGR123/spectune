#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECTUNE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRAJ_DIR="${TRAJ_DIR:-$SPECTUNE_ROOT/outputs/trajectories}"
ROLLOUT_PARQUET="${ROLLOUT_PARQUET:-$SPECTUNE_ROOT/outputs/datasets/verl/nmrexp_sft_rollout_qwen3_max_w_rs_2_topk_15.parquet}"
PORT_TRAJ="${PORT_TRAJ:-7860}"
PORT_STATS="${PORT_STATS:-7861}"
PORT_ROLLOUT="${PORT_ROLLOUT:-7862}"

cleanup() {
  echo "Stopping servers..."
  kill "$PID_TRAJ" "$PID_STATS" "$PID_ROLLOUT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 "$SCRIPT_DIR/rl_rollouts.py" --traj-dir "$TRAJ_DIR" --port "$PORT_TRAJ" &
PID_TRAJ=$!

python3 "$SCRIPT_DIR/rl_tool_stats.py" --traj-dir "$TRAJ_DIR" --port "$PORT_STATS" &
PID_STATS=$!

python3 "$SCRIPT_DIR/sft_rollouts.py" --parquet "$ROLLOUT_PARQUET" --port "$PORT_ROLLOUT" &
PID_ROLLOUT=$!

echo "Trajectory Viewer : http://localhost:$PORT_TRAJ"
echo "Tool Statistics   : http://localhost:$PORT_STATS"
echo "Rollout Viewer    : http://localhost:$PORT_ROLLOUT"

wait
