#!/usr/bin/env bash

# bash examples/debug/visualize.sh
# USE_NGROK=0 bash examples/debug/visualize.sh
# SERVICES=rl PYTHON_BIN=/path/to/python bash examples/debug/visualize.sh
# RL_STATS_MAX_FILES=0 bash examples/debug/visualize.sh  # scan every checkpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECTUNE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRAJ_DIR="${TRAJ_DIR:-$SPECTUNE_ROOT/outputs/trajectories/rl}"
EVAL_TRAJ_DIR="${EVAL_TRAJ_DIR:-$SPECTUNE_ROOT/outputs/trajectories/eval}"
SFT_PARQUET="${SFT_PARQUET:-$SPECTUNE_ROOT/outputs/datasets/verl/sft.parquet}"
PORT_TRAJ="${PORT_TRAJ:-7860}"
PORT_STATS="${PORT_STATS:-7861}"
PORT_ROLLOUT="${PORT_ROLLOUT:-7862}"
PORT_STATS_SFT="${PORT_STATS_SFT:-7863}"
PORT_EVAL="${PORT_EVAL:-7864}"
RL_STATS_MAX_FILES="${RL_STATS_MAX_FILES:-80}"

# all | rl | sft | eval | comma-separated: rl-rollouts,rl-stats,sft-rollouts,sft-stats,eval-rollouts
SERVICES="${SERVICES:-all}"

# Optional: set USE_NGROK=0 to disable ngrok tunnels
USE_NGROK="${USE_NGROK:-1}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3 "$SPECTUNE_ROOT/../envs/verl/bin/python"; do
    if { [[ -x "$candidate" ]] || command -v "$candidate" &>/dev/null; } \
       && "$candidate" -c "import flask, pandas, pyarrow" &>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: no Python environment with flask, pandas and pyarrow was found." >&2
  echo "Set PYTHON_BIN explicitly, for example:" >&2
  echo "  PYTHON_BIN=$SPECTUNE_ROOT/../envs/verl/bin/python bash $0" >&2
  exit 1
fi

NGROK_CONF=""
PID_NGROK=""
PIDS=()

service_enabled() {
  local name="$1"
  [[ "$SERVICES" == "all" ]] \
    || [[ ",$SERVICES," == *",$name,"* ]] \
    || [[ "$name" == rl-* && ",$SERVICES," == *",rl,"* ]] \
    || [[ "$name" == sft-* && ",$SERVICES," == *",sft,"* ]] \
    || [[ "$name" == eval-* && ",$SERVICES," == *",eval,"* ]]
}

start_service() {
  "$PYTHON_BIN" "$@" &
  PIDS+=("$!")
}

cleanup() {
  echo "Stopping servers..."
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
  [[ -n "$PID_NGROK" ]] && kill "$PID_NGROK" 2>/dev/null || true
  [[ -n "$NGROK_CONF" ]] && rm -f "$NGROK_CONF"
}
trap cleanup EXIT INT TERM

if service_enabled rl-rollouts; then
  start_service "$SCRIPT_DIR/rl_rollouts.py" --traj-dir "$TRAJ_DIR" --port "$PORT_TRAJ"
  echo "Trajectory Viewer : http://localhost:$PORT_TRAJ"
fi
if service_enabled rl-stats; then
  start_service "$SCRIPT_DIR/rl_tool_stats.py" --traj-dir "$TRAJ_DIR" --port "$PORT_STATS" \
    --max-files "$RL_STATS_MAX_FILES"
  echo "Tool Statistics   : http://localhost:$PORT_STATS (max files: $RL_STATS_MAX_FILES)"
fi
if service_enabled sft-rollouts; then
  start_service "$SCRIPT_DIR/sft_rollouts.py" --parquet "$SFT_PARQUET" --port "$PORT_ROLLOUT"
  echo "Rollout Viewer    : http://localhost:$PORT_ROLLOUT"
fi
if service_enabled sft-stats; then
  start_service "$SCRIPT_DIR/sft_stats.py" --parquet "$SFT_PARQUET" --port "$PORT_STATS_SFT"
  echo "SFT Stats         : http://localhost:$PORT_STATS_SFT"
fi
if service_enabled eval-rollouts; then
  start_service "$SCRIPT_DIR/eval_rollouts.py" --traj-dir "$EVAL_TRAJ_DIR" --port "$PORT_EVAL"
  echo "Eval Rollouts     : http://localhost:$PORT_EVAL"
fi

if ((${#PIDS[@]} == 0)); then
  echo "Error: SERVICES='$SERVICES' did not select any service." >&2
  exit 1
fi

if [[ "$USE_NGROK" == "1" ]] && command -v ngrok &>/dev/null; then
  NGROK_CONF=$(mktemp /tmp/ngrok-spectune-XXXXXX.yml)
  cat > "$NGROK_CONF" <<NGROK_EOF
version: "3"
tunnels:
NGROK_EOF
  TUNNEL_COUNT=0
  if service_enabled rl-rollouts; then
    cat >> "$NGROK_CONF" <<NGROK_EOF
  traj-viewer:
    proto: http
    addr: $PORT_TRAJ
NGROK_EOF
    ((TUNNEL_COUNT+=1))
  fi
  if service_enabled rl-stats; then
    cat >> "$NGROK_CONF" <<NGROK_EOF
  tool-stats:
    proto: http
    addr: $PORT_STATS
NGROK_EOF
    ((TUNNEL_COUNT+=1))
  fi
  if service_enabled sft-rollouts; then
    cat >> "$NGROK_CONF" <<NGROK_EOF
  rollout-viewer:
    proto: http
    addr: $PORT_ROLLOUT
NGROK_EOF
    ((TUNNEL_COUNT+=1))
  fi
  if service_enabled sft-stats; then
    cat >> "$NGROK_CONF" <<NGROK_EOF
  sft-stats:
    proto: http
    addr: $PORT_STATS_SFT
NGROK_EOF
    ((TUNNEL_COUNT+=1))
  fi
  if service_enabled eval-rollouts; then
    cat >> "$NGROK_CONF" <<NGROK_EOF
  eval-rollouts:
    proto: http
    addr: $PORT_EVAL
NGROK_EOF
    ((TUNNEL_COUNT+=1))
  fi

  ngrok start --all \
    --config="$HOME/.config/ngrok/ngrok.yml" \
    --config="$NGROK_CONF" \
    --log=stdout >/tmp/ngrok-spectune.log 2>&1 &
  PID_NGROK=$!

  echo ""
  echo "Waiting for ngrok tunnels..."
  for i in $(seq 1 15); do
    sleep 1
    TUNNELS=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null || true)
    COUNT=$(echo "$TUNNELS" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(len([t for t in data.get('tunnels', []) if t.get('public_url','').startswith('https')]))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
    if [[ "$COUNT" -ge "$TUNNEL_COUNT" ]]; then
      break
    fi
  done

  echo ""
  echo "=== Public URLs (ngrok) ==="
  curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    labels = {
        'traj-viewer':    'RL Rollouts Viewer',
        'tool-stats':     'RL Tool Statistics',
        'rollout-viewer': 'SFT Teacher Rollouts Viewer',
        'sft-stats':      'SFT Dataset Statistics',
        'eval-rollouts':  'Eval Rollouts Viewer',
    }
    for t in sorted(data.get('tunnels', []), key=lambda x: x.get('name','')):
        url = t.get('public_url', '')
        if url.startswith('https'):
            name = labels.get(t.get('name',''), t.get('name',''))
            print(f'{name}: {url}')
except Exception as e:
    print(f'(could not parse ngrok API: {e})')
" 2>/dev/null || echo "(ngrok API unavailable — check /tmp/ngrok-spectune.log)"
  echo ""
fi

wait -n "${PIDS[@]}"
