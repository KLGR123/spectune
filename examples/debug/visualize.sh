#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECTUNE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRAJ_DIR="${TRAJ_DIR:-$SPECTUNE_ROOT/outputs/trajectories}"
PORT_TRAJ="${PORT_TRAJ:-7860}"
PORT_STATS="${PORT_STATS:-7861}"

cleanup() {
  echo "Stopping servers..."
  kill "$PID_TRAJ" "$PID_STATS" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 "$SCRIPT_DIR/trajectories.py" --traj-dir "$TRAJ_DIR" --port "$PORT_TRAJ" &
PID_TRAJ=$!

python3 "$SCRIPT_DIR/tool_stats.py" --traj-dir "$TRAJ_DIR" --port "$PORT_STATS" &
PID_STATS=$!

echo "Trajectory Viewer : http://localhost:$PORT_TRAJ"
echo "Tool Statistics   : http://localhost:$PORT_STATS"

wait
