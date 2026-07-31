# Spectune via verl agentic RL

Spectune owns format contracts, artifact compilation, tools, and rewards.
verl is an optional adapter for Parquet I/O, multi-turn tool rollout, and
distributed GRPO / PPO. No Spectune core module imports torch / ray / verl.

## Layering

| Layer | Package | Responsibility |
|---|---|---|
| Contract | `spectune.format.v1` | hermes tool tags + `{"smiles":[...]}` answers |
| Catalog | `spectune.tools.catalog` | default RL tool names + live OpenAI schemas |
| Artifacts | `spectune.artifacts` | JSONL → versioned training rows |
| Reward | `spectune.reward` | framework-agnostic scoring |
| Tool adapter | `spectune.tools.verl` | `SpectuneTool` + tools YAML |
| Reward adapter | `spectune.reward.verl` | `compute_score` for verl reward managers |

## 1. Format contract (`v1`)

- Tools: hermes `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- Final answer: text **after** the last tool call, as `{"smiles": ["...", "..."]}`
- System prompt: `spectune.format.v1.SYSTEM_PROMPT` (injected by the artifact compiler)

Decoded verl response strings often glue tool-response tokens into the same
flat string. Spectune rebuilds assistant / tool turns and scores only the final
answer region (preferring the last `{"smiles":[...]}` object).

## 2. Compile JSONL → training artifacts

```bash
cd spectune
pip install -e '.[dev,data,chem]'

python -m spectune.artifacts compile \
  --input outputs/datasets/nmrexp_train_20000.jsonl \
  --output outputs/verl/train.parquet \
  --split train \
  --all-tools \
  --manifest outputs/verl/train.manifest.json

python -m spectune.artifacts compile \
  --input outputs/datasets/nmrexp_test.jsonl \
  --output outputs/verl/test.parquet \
  --split test \
  --all-tools
```

`--all-tools` expands to `DEFAULT_RL_TOOL_NAMES`. Each row includes:

- `agent_name=tool_agent`
- `prompt` (system + user turns)
- `reward_model.ground_truth`
- `extra_info.tool_names` / `tools_kwargs`

Full `tool_schemas` are **not** embedded by default (Parquet stays lean). The
reward adapter resolves live schemas from `tool_names` via `ToolManager`. Pass
`--include-tool-schemas` only when the scoring host cannot import Spectune tools.

## 3. Emit verl tool YAML

```bash
python - <<'PY'
from pathlib import Path
from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES
from spectune.tools.verl import write_tools_config

write_tools_config(
    Path("examples/verl/tools_config.yaml"),
    tool_names=DEFAULT_RL_TOOL_NAMES,
)
print("wrote examples/verl/tools_config.yaml")
PY
```

`SpectuneTool` loads **full** schemas from `ToolManager` (it does not use verl's
pydantic schema models, which would drop `items` / `minimum` / …). Keep YAML
entries schema-free (`tool_name` only).

## 4. Point verl at Spectune reward + tools

```bash
export SPECTUNE_ROOT=/path/to/spectune
export VERL_ROOT=/path/to/verl
export TRAIN_FILE=$SPECTUNE_ROOT/outputs/verl/train.parquet
export TEST_FILE=$SPECTUNE_ROOT/outputs/verl/test.parquet
export TOOL_CONFIG=$SPECTUNE_ROOT/examples/verl/tools_config.yaml

bash $SPECTUNE_ROOT/examples/verl/run_grpo_spectune.sh
```

Key Hydra overrides:

- `reward.custom_reward_function.path=pkg://spectune.reward.verl`
- `reward.custom_reward_function.name=compute_score`
- `actor_rollout_ref.rollout.multi_turn.enable=True`
- `actor_rollout_ref.rollout.multi_turn.format=hermes`
- `actor_rollout_ref.rollout.multi_turn.tool_config_path=...`
- `actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent`

`data.tool_config_path` mirrors the rollout tool config automatically in current
verl releases, so prompt-length filtering sees the same schemas as rollout.
