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

Putting `{name}` (and optionally `{size}`) in `--output` builds the train/test/sft
sets in a single pass: the truth pool is shuffled once with `--seed`, then carved
into three non-overlapping slices, so disjointness is guaranteed by construction.

```bash
cd /path/to/spectune

python -m spectune.augmentor \
    --output     outputs/datasets/nmrexp_{name}_{size}.jsonl \
    --split      train \
    --sizes      20000 200 2000 \
    --information-mix '{"none":0.24,"formula":0.24,"structure":0.04,"reaction":0.24,"fragment":0.24}' \
    --followup-probability 0.5 \
    --max-fragment-items 3 \
    --max-reaction-items 3 \
    --nmr-noise-ratio 0.0 \
    --formula-noise-ratio 0.0 \
    --modal-drop-ratio 0.5
```

writes three mutually disjoint files, all drawn from the `train` truth split:

- `outputs/datasets/nmrexp_train_20000.jsonl` — RL train set
- `outputs/datasets/nmrexp_test_200.jsonl` — held-out eval set
- `outputs/datasets/nmrexp_sft_2000.jsonl` — SFT set

A plain `--output` without `{name}` keeps the original single-file behavior:

```bash
python -m spectune.augmentor \
    --output      outputs/datasets/nmrexp_train_20000.jsonl \
    --split       train \
    --sample-size 20000 \
    --information-mix '{"none":0.24,"formula":0.24,"structure":0.04,"reaction":0.24,"fragment":0.24}' \
    --followup-probability 0.5
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--output` | *(required)* | Destination JSONL path; `{name}` / `{size}` placeholders switch to multi-split mode |
| `--sizes` | `20000 200 2000` | Train/test/sft row counts in multi-split mode |
| `--split` | `train` | Dataset split to draw from |
| `--sample-size` | `0` (full split) | Rows to sample in single-file mode |
| `--seed` | `42` | Random seed (also orders the disjoint slices) |
| `--information-mix` | `none`30% `formula`20% `structure`20% `reaction`15% `fragment`15% | JSON dict, shares must sum to 1 |
| `--followup-probability` | `0.4` | Share where extra info arrives as a second turn |
| `--max-fragment-items` | `2` | Fragment hints per sample |
| `--max-reaction-items` | `2` | Reaction context items per sample |
| `--nmr-noise-ratio` | `0.0` | Share of rows with corrupted NMR spectrum |
| `--use-llm-rewrite` | off | Rewrite queries with LLM |

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
  --manifest outputs/datasets/verl/train.manifest.json \
  --interaction-config outputs/datasets/verl/interaction_config.yaml

python -m spectune.artifacts compile \
  --input outputs/datasets/nmrexp_test_200.jsonl \
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

# raise the nmr_generate candidate budget (default topk 10, max 50)
python -m spectune.tools write-config \
    --output outputs/datasets/verl/tools_config.yaml \
    --nmr-gen-topk 30
```

`--nmr-gen-topk` (default `10`, max `50`) sets the default `topk` of the
`nmr_generate` tool; it is recorded on the `nmr_generate` entry and honored by
`SpectuneTool` for both the tool schema default and tool execution during GRPO
rollout.

`SpectuneTool` loads **full** schemas from `ToolManager` (it does not use verl's
pydantic schema models, which would drop `items` / `minimum` / …). Keep YAML
entries schema-free (`tool_name` only).

## SFT with Rollout-Filtered Data

Supervised fine-tuning using rejection-sampled rollouts as training examples.

Runs a hermes tool-agent loop per sample via `spectune.rollout`: the LLM generates,
`<tool_call>` blocks execute for real via `ToolManager`, and generation repeats until
the model stops calling tools or `--max-assistant-turns` is reached. Each trajectory
retries up to `--rounds` times, keeping only the first one where the model identifies
the ground-truth SMILES (GT rejection sampling via `spectune.reward.RewardEvaluator`).
Each accepted record is a `messages` list ready for verl multiturn SFT.

```bash
cd /path/to/spectune

# API mode: SPECTUNE_LLM_BASE_URL / SPECTUNE_LLM_MODEL from secrets.env
python -m spectune.rollout \
    --input   outputs/datasets/nmrexp_sft_2000.jsonl \
    --output  outputs/datasets/verl/nmrexp_sft_rollout_qwen3_max_w_rs_2_topk_15.parquet \
    --format  parquet \
    --rounds  2 \
    --temperature 0.5 \
    --max-tokens 10000 \
    --nmr-gen-topk 15 \
    --max-concurrency 8 \
    --model qwen3-max

# Local mode: pass --local-model (or set SPECTUNE_LOCAL_MODEL_PATH in secrets.env) to
# have spectune auto-start a vLLM OpenAI-compatible server, run rollout against it, and
# shut it down when done. SPECTUNE_LLM_BASE_URL / SPECTUNE_LLM_MODEL are ignored here.
python -m spectune.rollout \
    --input              outputs/datasets/nmrexp_sft_2000.jsonl \
    --output             outputs/datasets/verl/nmrexp_sft_rollout_qwen_32b_w_rs_2_topk_15.parquet \
    --format             parquet \
    --rounds             2 \
    --temperature        0.5 \
    --max-tokens         8000 \
    --nmr-gen-topk       15 \
    --max-concurrency    6 \
    --local-model        /fs_mol/liujiarun/models/qwen3-32b \
    --tensor-parallel-size 8 \
    --max-model-len      20000 \
    --gpu-memory-utilization 0.85
```

`--max-concurrency` caps in-flight LLM requests in both modes; for local vLLM it also
sets the request-queue depth (too low starves continuous batching, too high causes
memory pressure) — 4–8 is a reasonable starting point for a single node.

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | nmrexp_sft_*.jsonl from `python -m spectune.augmentor` |
| `--output` | auto-derived | Destination JSONL or Parquet |
| `--format` | `jsonl` | Output format (`jsonl` / `parquet`) |
| `--rounds` | `8` | Max rejection-sampling rounds per sample (k) |
| `--temperature` | `0.8` | LLM sampling temperature |
| `--top-p` | `0.95` | Top-p nucleus sampling |
| `--max-tokens` | `4096` | Max tokens per LLM response |
| `--nmr-gen-topk` | `10` | Default `topk` for `nmr_generate` candidate generation (max `50`) |
| `--tools` | `DEFAULT_RL_TOOL_NAMES` | Tool names exposed to the model |
| `--max-assistant-turns` | `16` | Max LLM generations per agent loop (matches verl `multi_turn.max_assistant_turns`) |
| `--max-concurrency` | `8` | Max concurrent LLM requests |
| `--model` | `SPECTUNE_LLM_MODEL` env | Override LLM model name (e.g. `GPT-5.4`) |
| `--local-model` | `SPECTUNE_LOCAL_MODEL_PATH` | Path to local HuggingFace model directory; triggers auto vLLM server |
| `--tensor-parallel-size` | `SPECTUNE_VLLM_TENSOR_PARALLEL_SIZE` (or `1`) | GPUs for tensor parallelism (local mode) |
| `--vllm-port` | random free port | Fixed TCP port for the vLLM server (local mode, useful for debugging) |
| `--no-progress` | off | Suppress tqdm progress bar |

LLM endpoint is read from `SPECTUNE_LLM_BASE_URL`, `SPECTUNE_LLM_MODEL`, and
`SPECTUNE_LLM_API_KEY` (see `secrets.env.example`).

Then Launch SFT.

```bash
bash examples/grpo/run_sft.sh
```

Calls `verl.trainer.fsdp_sft_trainer` via `torchrun` with multiturn mode enabled
(`data.multiturn.messages_key=messages`). Key overrides:

- `data.multiturn.enable=true` — treat each row's `messages` list as a full conversation
- `data.max_length=8192` — covers tool-call traces
- `model.partial_pretrain` — base model path (default: `qwen3-8b`)
- `trainer.total_epochs=3` — adjust to dataset size

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

## Visualise rollout trajectories

Two viewers are available; use different ports to run them side-by-side:

| Viewer | Script | Default port | Data source |
|--------|--------|-------------|-------------|
| GRPO rollout (JSONL dir) | `examples/debug/traj_rl.py` | 7860 | `outputs/trajectories/` |
| SFT rollout (Parquet) | `examples/debug/traj_sft.py` | 7861 | `outputs/datasets/verl/*.parquet` |

```bash
# Start all three viewers at once (ports 7860 / 7861 / 7862)
cd /path/to/spectune
bash examples/debug/visualize.sh

# Override the rollout parquet file or ports via env vars
ROLLOUT_PARQUET=outputs/datasets/verl/nmrexp_sft_rollout_qwen3_max_w_rs_2_topk_15.parquet \
PORT_ROLLOUT=7862 \
bash examples/debug/visualize.sh
```
