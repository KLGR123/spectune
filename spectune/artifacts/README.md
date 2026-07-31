# Spectune training artifacts

Compile Spectune augmentor JSONL (`turns` + `gt_smiles`) into versioned rows
that external RL backends can consume without importing Spectune's training
loop.

```bash
python -m spectune.artifacts compile \
  --input outputs/datasets/nmrexp_train_20000.jsonl \
  --output outputs/verl/train.parquet \
  --split train \
  --all-tools
```

Each row contains:

- `prompt`: system (`format.v1.SYSTEM_PROMPT`) + user turns
- `agent_name`: `tool_agent`
- `reward_model.ground_truth`: `gt_smiles`
- `extra_info.format_spec` / `tool_names` / `tools_kwargs` / optional `reward_config`

Full OpenAI `tool_schemas` are omitted by default so Parquet stays small; the
reward adapter resolves schemas from `tool_names` at scoring time. Pass
`--include-tool-schemas` only when the scoring host cannot import Spectune.

Parquet output requires `pip install spectune[data]`. JSONL output needs no
extras.
