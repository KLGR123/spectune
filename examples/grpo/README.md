## Preprocess NMRexp raw data

Merges rows with the same canonical SMILES across all sources within each split.

```bash
cd /path/to/spectune

# Build both train and test truth files (default)
python -m spectune.dataloader preprocess \
    --raw-dir      /path/to/data/NMRexp \
    --datasets-dir outputs/datasets

# Build only the train split, cap at 100k records
python -m spectune.dataloader preprocess \
    --splits       train \
    --raw-dir      /path/to/data/NMRexp \
    --datasets-dir outputs/datasets \
    --max-records  100000

# Check sizes without rebuilding
python -m spectune.dataloader info \
    --datasets-dir outputs/datasets
```

Key flags for `preprocess`:

| Flag | Default | Description |
|---|---|---|
| `--splits` | all | Which splits to build (`train` / `test`) |
| `--raw-dir` | `NMREXP_RAW_DIR` | Directory with raw NMRexp exports |
| `--datasets-dir` | `SPECTUNE_DATASETS_DIR` | Output directory for JSONL files |
| `--max-records` | `0` (unlimited) | Cap total records per split after merging |
| `--allowed-nmr-types` | all | Whitelist of NMR types, e.g. `'1H NMR'` `'13C NMR'` |
| `--min-quality` | `any` | QA gate for checked sources (`any` / `same_skeleton` / `same_molecule`) |
| `--overwrite` | off | Rebuild even if cache exists |
| `--no-progress` | off | Suppress per-row progress |

## Cluster truth records

Annotates each record with a `cluster` label (MiniBatchKMeans over Morgan fingerprints + molecular properties). Rewrites the JSONL in place.

```bash
python -m spectune.classifier \
    outputs/datasets/nmrexp_truth_train.jsonl \
    --clusters  256 \
    --workers   16 \
    --visualize \
    --visualization-path outputs/datasets/train_clusters.png

python -m spectune.classifier \
    outputs/datasets/nmrexp_truth_test.jsonl \
    --clusters  64 \
    --workers   8
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--clusters` | `256` | Number of KMeans clusters |
| `--workers` | `min(16, cpu_count)` | Parallel RDKit featurization workers |
| `--seed` | `42` | Random seed |
| `--batch-size` | `2048` | MiniBatchKMeans batch size |
| `--epochs` | `2` | Training passes over the data |
| `--visualize` | off | Save a cluster scatter-plot PNG |
| `--visualization-path` | auto | Override PNG output path |
| `--visualization-dim` | `2` | `2` or `3` for the scatter-plot |

## Build augmented dataset

```bash
cd /path/to/spectune

python -m spectune.augmentor \
    --output     outputs/datasets/nmrexp_train_20000.jsonl \
    --split      train \
    --sample-size 20000 \
    --information-mix '{"none":0.24,"formula":0.24,"structure":0.04,"reaction":0.24,"fragment":0.24}' \
    --followup-probability 0.5 \
    --max-fragment-items 3 \
    --max-reaction-items 3 \
    --nmr-noise-ratio 0.0 \
    --formula-noise-ratio 0.0 \
    --modal-drop-ratio 0.5

```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--output` | *(required)* | Destination JSONL file |
| `--split` | `train` | Dataset split |
| `--sample-size` | `0` (full split) | Rows to sample |
| `--seed` | `42` | Random seed |
| `--information-mix` | `none`30% `formula`20% `structure`20% `reaction`15% `fragment`15% | JSON dict, shares must sum to 1 |
| `--followup-probability` | `0.4` | Share where extra info arrives as a second turn |
| `--max-fragment-items` | `2` | Fragment hints per sample |
| `--max-reaction-items` | `2` | Reaction context items per sample |
| `--nmr-noise-ratio` | `0.0` | Share of rows with corrupted NMR spectrum |
| `--use-llm-rewrite` | off | Rewrite queries with LLM |

See `examples/grpo/augmentation.py` for the config used in the `grpo-nmrexp-20k-qwen3-4b-base` experiment.

## Format contract (`v1`)

- Tools: hermes `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- Final answer: text **after** the last tool call, as `{"smiles": ["...", "..."]}`
- System prompt: `spectune.format.v1.SYSTEM_PROMPT` (injected by the artifact compiler)

Decoded verl response strings often glue tool-response tokens into the same
flat string. Spectune rebuilds assistant / tool turns and scores only the final
answer region (preferring the last `{"smiles":[...]}` object).

## Compile JSONL to training artifacts

```bash
cd spectune
pip install -e '.[dev,data,chem]'

python -m spectune.artifacts compile \
  --input outputs/datasets/nmrexp_train_20000.jsonl \
  --output outputs/datasets/verl/train.parquet \
  --split train \
  --manifest outputs/datasets/verl/train.manifest.json

python -m spectune.artifacts compile \
  --input outputs/datasets/nmrexp_test.jsonl \
  --output outputs/datasets/verl/test.parquet \
  --split test
```

`--tools` accepts a comma-separated list of tool names; omit it to use `DEFAULT_RL_TOOL_NAMES`. Each row includes:

- `agent_name=tool_agent`
- `prompt` (system + user turns)
- `reward_model.ground_truth`
- `extra_info.tool_names` / `tools_kwargs`

Full `tool_schemas` are **not** embedded by default (Parquet stays lean). The
reward adapter resolves live schemas from `tool_names` via `ToolManager`. Pass
`--include-tool-schemas` only when the scoring host cannot import Spectune tools.

## Emit verl tool YAML

```bash
cd /path/to/spectune

python -m spectune.tools write-config \
    --output outputs/datasets/verl/tools_config.yaml

# restrict to a subset
python -m spectune.tools write-config \
    --output outputs/datasets/verl/tools_config.yaml \
    --tools nmr_generate nmr_repair nmr_forward_predict
```

`SpectuneTool` loads **full** schemas from `ToolManager` (it does not use verl's
pydantic schema models, which would drop `items` / `minimum` / …). Keep YAML
entries schema-free (`tool_name` only).

## Point verl at Spectune reward + tools

```bash
bash /path/to/spectune/examples/grpo/run_grpo.sh
```

Key Hydra overrides:

- `custom_reward_function.path=$SPECTUNE_ROOT/spectune/reward/verl.py`  
- `custom_reward_function.name=compute_score`
- `reward_model.reward_manager=naive` (async agent loop only registers `naive` / `dapo`)
- `actor_rollout_ref.rollout.multi_turn.enable=True`
- `actor_rollout_ref.rollout.multi_turn.format=hermes`
- `actor_rollout_ref.rollout.multi_turn.tool_config_path=...`
- `actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent`

`data.tool_config_path` mirrors the rollout tool config automatically in current
verl releases, so prompt-length filtering sees the same schemas as rollout.

## Merge Model Shards

```bash
cd /path/to/verl

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/checkpoints/spectune/grpo-nmrexp-20k-qwen3-4b-base/global_step_500/actor \
    --target_dir /path/to/checkpoints/spectune/grpo-nmrexp-20k-qwen3-4b-base/global_step_500/actor/huggingface
```
