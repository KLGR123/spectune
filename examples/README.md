## Preprocess NMRexp raw data

Merges rows with the same canonical SMILES across all sources within each split.

```bash
cd /path/to/spectune

# Build both truth files (default): "bulk" (large, unverified) and
# "verified" (small, human-checked) -- named by provenance, not by
# train/test role (see note below).
python -m spectune.dataloader preprocess \
    --raw-dir      /path/to/data/NMRexp \
    --datasets-dir outputs/datasets

# Build only the bulk split, cap at 100k records
python -m spectune.dataloader preprocess \
    --splits       bulk \
    --raw-dir      /path/to/data/NMRexp \
    --datasets-dir outputs/datasets \
    --max-records  100000

# Check sizes without rebuilding
python -m spectune.dataloader info \
    --datasets-dir outputs/datasets
```

> **Note on naming**: these truth splits (`bulk`/`verified`) describe where a
> row came from, not whether it ends up in a train/test/sft output file.
> `spectune.augmentor` below draws *all* of its train/test/sft output splits
> from a single truth split (`--split bulk` by default), so a `bulk`-sourced
> row can land in any of the three output files -- deliberately kept separate
> from the truth-split name to avoid a `sample_id` that reads "train" even
> when the row was sampled into the held-out eval set.

Key flags for `preprocess`:

| Flag | Default | Description |
|---|---|---|
| `--splits` | all | Which splits to build (`bulk` / `verified`) |
| `--raw-dir` | `/root/data/NMRexp` | Directory with raw NMRexp exports |
| `--datasets-dir` | `outputs/datasets` | Output directory for JSONL files |
| `--max-records` | `0` (unlimited) | Cap total records per split after merging |
| `--allowed-nmr-types` | all | Whitelist of NMR types, e.g. `'1H NMR'` `'13C NMR'` |
| `--min-quality` | `any` | QA gate for checked sources (`any` / `same_skeleton` / `same_molecule`) |
| `--overwrite` | off | Rebuild even if cache exists |
| `--no-progress` | off | Suppress per-row progress |

## Cluster truth records

Annotates each record with a `cluster` label (MiniBatchKMeans over Morgan fingerprints + molecular properties). Rewrites the JSONL in place.

```bash
python -m spectune.classifier \
    outputs/datasets/nmrexp_truth_bulk.jsonl \
    --clusters  256 \
    --workers   16 \
    --visualize \
    --visualization-path outputs/bulk_clusters.png

python -m spectune.classifier \
    outputs/datasets/nmrexp_truth_verified.jsonl \
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
    --split      bulk \
    --sizes      20000 200 2000 \
    --information-mix '{"none":0.24,"formula":0.24,"structure":0.04,"reaction":0.24,"fragment":0.24}' \
    --followup-probability 0.5 \
    --max-fragment-items 3 \
    --max-reaction-items 3 \
    --nmr-noise-ratio 0.0 \
    --formula-noise-ratio 0.0 \
    --modal-drop-ratio 0.5
```

writes three mutually disjoint files, all drawn from the `bulk` truth split
(so every `sample_id` below carries a `NMRexp:bulk:...` prefix regardless of
which output file it lands in -- that prefix names the *source* truth split,
not the output role):

- `outputs/datasets/nmrexp_train_20000.jsonl` — RL train set
- `outputs/datasets/nmrexp_test_200.jsonl` — held-out eval set
- `outputs/datasets/nmrexp_sft_2000.jsonl` — SFT set

A plain `--output` without `{name}` keeps the original single-file behavior:

```bash
python -m spectune.augmentor \
    --output      outputs/datasets/nmrexp_rl_20000.jsonl \
    --split       bulk \
    --sample-size 20000 \
    --information-mix '{"none":0.24,"formula":0.24,"structure":0.04,"reaction":0.24,"fragment":0.24}' \
    --followup-probability 0.5
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--output` | *(required)* | Destination JSONL path; `{name}` / `{size}` placeholders switch to multi-split mode |
| `--sizes` | `20000 200 2000` | Train/test/sft row counts in multi-split mode |
| `--split` | `bulk` | Truth split to draw from (`bulk` / `verified`); unrelated to the output train/test/sft split names |
| `--sample-size` | `0` (full split) | Rows to sample in single-file mode |
| `--seed` | `42` | Random seed (also orders the disjoint slices) |
| `--datasets-dir` | `outputs/datasets` | Dataloader truth cache + enrichment cache directory; independent of `--output` |
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
  --input outputs/datasets/nmrexp_rl_20000.jsonl outputs/datasets/others_rl_23861.jsonl \
  --data-source spectune/nmrexp spectune/others \
  --output outputs/datasets/verl/train.parquet \
  --split train \
  --manifest outputs/datasets/verl/train.manifest.json \
  --interaction-config outputs/datasets/verl/interaction_config.yaml

python -m spectune.artifacts compile \
  --input outputs/datasets/nmrexp_test_200.jsonl outputs/datasets/others_test_2600.jsonl \
  --output outputs/datasets/verl/test.parquet \
  --split test
```

The `others_*.jsonl` files are not produced by this repo's `dataloader`/
`augmentor`; they come from a separate preprocessing pipeline
(`/path/to/data/SFT/scripts/{preprocess,convert}.py`) that normalizes
several heterogeneous SFT sources (`fragment`, `structure`, `reaction`,
`formula`, `none`) into the same `{sample_id, gt_smiles, turns}` shape
`compile` expects, then splits the result into an RL pool and a held-out
pre-SFT pool (`others_rl_23861.jsonl` + `others_pre_sft_2000.jsonl`, disjoint,
25861 rows combined) plus a separate `others_test_2600.jsonl`. Their
`sample_id` encodes the task type as the first colon-delimited segment (e.g.
`fragment:train:1778`), unlike `NMRexp:...:aug` ids, which carry the type in
`augmentation.information_type` instead. `compile` resolves both conventions
into a single canonical `extra_info.data_type` field at compile time — see
`infer_data_type` in [`spectune/artifacts/compile.py`](../spectune/artifacts/compile.py)
(also used directly on raw JSONL by
[`spectune/rollout/io.py`](../spectune/rollout/io.py) for
`python -m spectune.rollout.eval`). The `sample_id` prefix must be an exact
`INFORMATION_TYPES` match; the original preprocessing pipeline emitted an
unresolved compound prefix (`none_plus_formula`, mixing rows that stated a
molecular formula with rows that didn't), which was split by checking
whether the RDKit-computed formula for `gt_smiles` appears verbatim in the
user turn text, then relabeled to `formula:...` / `none:...` in place.

`--tools` takes one or more space-separated tool names (e.g. `--tools nmr_generate nmr_repair`); omit it to use `DEFAULT_RL_TOOL_NAMES`. `--data-source` takes either one value (applied to every `--input` path) or exactly one value per `--input` path, so the NMRexp and "others" pools above can be tagged `spectune/nmrexp` / `spectune/others` while compiling in one run instead of two. Each row includes:

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
    --nmr-gen-topk 15
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

Progress is checkpointed incrementally to `<output>.checkpoint.jsonl`; a crashed run
resumes automatically from where it left off on the next invocation.  Pass
`--no-checkpoint` to disable.

Three backends are supported. HTTP mode.

```bash
cd /path/to/spectune

export SPECTUNE_LLM_BASE_URL=http://your-api-host/v1
export SPECTUNE_LLM_MODEL=qwen3-max

python -m spectune.rollout \
    --input        outputs/datasets/nmrexp_pre_sft_2000.jsonl \
    --output       outputs/datasets/verl/nmrexp_sft_rollout_qwen3_max_w_rs_2_topk_15.parquet \
    --format       parquet \
    --rounds       2 \
    --temperature  0.5 \
    --max-tokens   10000 \
    --nmr-gen-topk 15 \
    --max-concurrency 8
```

`--input` accepts multiple JSONL files; when more than one is given, all
samples are combined and shuffled together before rollout.

```bash
python -m spectune.rollout \
    --input        outputs/datasets/nmrexp_pre_sft_2000.jsonl outputs/datasets/others_pre_sft_2000.jsonl \
    --output       outputs/datasets/verl/nmrexp_sft_rollout_qwen3_max_w_rs_2_topk_15.parquet \
    --format       jsonl \
    --rounds       2 \
    --temperature  0.5 \
    --max-tokens   10000 \
    --nmr-gen-topk 15 \
    --max-concurrency 8
```

`--skills` accepts one or more skill files (default: `.md`); their contents
are appended to the end of the system prompt in order, each separated by a
newline. Off by default.

```bash
python -m spectune.rollout \
    --input        outputs/datasets/nmrexp_sft_2000.jsonl \
    --output       outputs/datasets/sft/nmrexp_sft_rollout_qwen3_max_w_rs_2_topk_15.parquet \
    --format       parquet \
    --rounds       2 \
    --temperature  0.5 \
    --max-tokens   10000 \
    --nmr-gen-topk 15 \
    --max-concurrency 6 \
    --skills       spectune/format/rdkit.md
```

litellm mode.

```bash
python -m spectune.rollout \
    --backend      litellm \
    --input        outputs/datasets/nmrexp_pre_sft_2000.jsonl outputs/datasets/others_pre_sft_2000.jsonl \
    --output       outputs/datasets/sft/nmrexp_sft_rollout_claude_sonnet_46_w_rs_1_topk_15.parquet \
    --format       jsonl \
    --rounds       1 \
    --temperature  0.3 \
    --max-tokens   25600 \
    --nmr-gen-topk 15 \
    --max-concurrency 4 \
    --model        openai/claude-sonnet-4-6 \
    --skills       spectune/format/rdkit.md
```

Local vLLM mode.

```bash
python -m spectune.rollout \
    --backend                local \
    --model                  /path/to/models/qwen3-32b \
    --input                  outputs/datasets/nmrexp_sft_2000.jsonl \
    --output                 outputs/datasets/sft/nmrexp_sft_rollout_qwen_32b_w_rs_2_topk_15.parquet \
    --format                 parquet \
    --rounds                 2 \
    --temperature            0.5 \
    --max-tokens             8000 \
    --nmr-gen-topk           15 \
    --max-concurrency        6 \
    --tensor-parallel-size   8 \
    --max-model-len          20000 \
    --gpu-memory-utilization 0.85
```

`--max-concurrency` caps in-flight LLM requests in all modes; for local vLLM it also
sets the request-queue depth (too low starves continuous batching, too high causes
memory pressure) — 4–8 is a reasonable starting point for a single node.

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | nmrexp_sft_*.jsonl from `python -m spectune.augmentor`; accepts multiple files, whose samples are combined and shuffled together when more than one is given |
| `--output` | auto-derived | Destination JSONL or Parquet |
| `--format` | `jsonl` | Output format (`jsonl` / `parquet`) |
| `--rounds` | `None` (no RS) | Max rejection-sampling rounds per sample; omit to accept every trajectory unconditionally |
| `--temperature` | `0.8` | LLM sampling temperature |
| `--top-p` | `0.95` | Top-p nucleus sampling |
| `--max-tokens` | `4096` | Max tokens per LLM response |
| `--nmr-gen-topk` | `10` | Default `topk` for `nmr_generate` candidate generation (max `50`) |
| `--tools` | `DEFAULT_RL_TOOL_NAMES` | Tool names exposed to the model |
| `--max-assistant-turns` | `16` | Max LLM generations per agent loop (matches verl `multi_turn.max_assistant_turns`) |
| `--max-concurrency` | `8` | Max concurrent LLM requests |
| `--skills` | off | Skill file(s) (default: `.md`) appended to the end of the system prompt, one per file separated by a newline |
| `--backend` | `http` | LLM backend: `http` (direct OpenAI-compatible), `litellm` (100+ providers), or `local` (auto-starts vLLM) |
| `--model` | env var (`http`/`litellm`) or *(required)* (`local`) | Model name or path; for `http` reads `SPECTUNE_LLM_MODEL`, for `litellm` reads `LITELLM_MODEL`, for `local` is the HuggingFace model directory path -- no env var fallback, always pass it explicitly |
| `--tensor-parallel-size` | `1` | GPUs for vLLM tensor parallelism (`--backend local` only) |
| `--vllm-port` | random free port | Fixed TCP port for the vLLM server (`--backend local` only, useful for debugging) |
| `--max-model-len` | model config | KV-cache sequence length cap passed to vLLM; set equal to `--max-tokens` to avoid OOM during CUDA graph capture (`--backend local` only) |
| `--gpu-memory-utilization` | `0.90` | Fraction of GPU memory vLLM may use; lower to `0.85` to leave headroom (`--backend local` only) |
| `--no-checkpoint` | off | Disable incremental checkpoint saves (enabled by default) |
| `--no-progress` | off | Suppress per-sample progress output |

HTTP endpoint env vars: `SPECTUNE_LLM_BASE_URL`, `SPECTUNE_LLM_MODEL`, `SPECTUNE_LLM_API_KEY`.  
litellm env vars: `LITELLM_API_BASE`, `LITELLM_MODEL`, `LITELLM_API_KEY`.  
local backend has no env vars: pass `--model /path/to/weights [--tensor-parallel-size N]` on the command line every time -- which local model to serve is a per-run choice, not something to bake into `secrets.env`.  
(See `secrets.env.example` for the full list.)

## Post-process Rollout Dumps into a SFT Parquet

Merge and filter one or more rollout JSONL dumps into a single
`train_sft.parquet` ready for verl SFT. Records are kept only when their
`reward_score` exceeds `--min-reward`. Optionally, conversations whose rendered
chat template exceeds the SFT context limit can be dropped before survivors are
shuffled and written as a single flat Parquet file.

```bash
cd /path/to/spectune

# --input takes one or more JSONL paths (shell-glob outputs/datasets/sft/*.jsonl
# to pick up every dump in that directory), same style as --input elsewhere
# (spectune.rollout, spectune.rollout.eval, spectune.artifacts compile).
python -m spectune.rollout.postprocess \
    --input      outputs/datasets/sft/*.jsonl \
    --output     outputs/datasets/verl/sft.parquet \
    --min-reward 0.1 \
    --seed       42

# Also remove samples longer than the SFT context window
python -m spectune.rollout.postprocess \
    --input         outputs/datasets/sft/*.jsonl \
    --output        outputs/datasets/verl/sft.parquet \
    --min-reward    0.1 \
    --drop-overlong \
    --tokenizer     /path/to/models/qwen3-8b \
    --max-length    32768
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | One or more rollout JSONL dump paths (e.g. shell-glob `outputs/datasets/sft/*.jsonl`); all records are merged before filtering |
| `--output` | `outputs/datasets/verl/train_sft.parquet` | Destination Parquet file |
| `--min-reward` | `0.1` | Minimum `reward_score` to keep a record |
| `--seed` | `42` | Random seed for shuffling |
| `--drop-overlong` | off | Drop records whose chat-template token count exceeds `--max-length` |
| `--tokenizer` | none | Hugging Face tokenizer name or local path; required with `--drop-overlong` |
| `--max-length` | `32768` | Maximum rendered conversation length used by `--drop-overlong` |
| `--no-progress` | off | Suppress per-file progress output |

Then Launch SFT.

```bash
bash examples/run_sft.sh
```

Calls `verl.trainer.fsdp_sft_trainer` via `torchrun` with multiturn mode enabled
(`data.multiturn.messages_key=messages`). Key overrides:

- `data.multiturn.enable=true` — treat each row's `messages` list as a full conversation
- `data.max_length=8192` — covers tool-call traces
- `model.partial_pretrain` — base model path (default: `qwen3-8b`)
- `trainer.total_epochs=3` — adjust to dataset size

## Point verl at Spectune reward + tools

```bash
bash /path/to/spectune/examples/run_grpo.sh
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

Four viewers are available; `examples/debug/visualize.sh` runs them all
side-by-side on different ports:

| Viewer | Script | Default port | Data source |
|--------|--------|-------------|-------------|
| GRPO rollout trajectories (JSONL dir) | `examples/debug/rl_rollouts.py` | 7860 | `outputs/trajectories/spectune/` |
| GRPO tool-call statistics | `examples/debug/rl_tool_stats.py` | 7861 | `outputs/trajectories/spectune/` |
| SFT teacher rollouts (Parquet dir) | `examples/debug/sft_rollouts.py` | 7862 | `outputs/datasets/sft/` |
| SFT dataset statistics (single Parquet) | `examples/debug/sft_stats.py` | 7863 | `outputs/datasets/verl/train_sft.parquet` |

```bash
# Start all four viewers at once (ports 7860-7863)
cd /path/to/spectune
bash examples/debug/visualize.sh

# Override data sources or ports via env vars
TRAJ_DIR=outputs/trajectories/my-run \
SFT_PARQUET=outputs/datasets/verl/train_sft_overlong_filtered.parquet \
PORT_STATS_SFT=7864 \
bash examples/debug/visualize.sh
```

## Evaluate a Local Merged Model

`python -m spectune.rollout.eval` shares its CLI flags and LLM-backend
dispatch with `python -m spectune.rollout` (see [`spectune/rollout/cli.py`](../spectune/rollout/cli.py)),
and reads samples the same way (see [`spectune/rollout/io.py`](../spectune/rollout/io.py)).
It runs the same offline tool-agent rollout, then reports hit@k metrics
overall and broken down by `data_type` (the canonical scenario label written
by `spectune.artifacts.compile`, or resolved on the fly for raw JSONL input).

```bash
cd /path/to/spectune

python -m spectune.rollout.eval \
    --input        outputs/datasets/verl/test.parquet \
    --output       outputs/trajectories/eval/test-20260831.jsonl \
    --backend      local \
    --model        outputs/checkpoints/spectune/sft-all-rollout-qwen3-8b/global_step_216/huggingface \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --max-tokens   8192 \
    --temperature  0.6 \
    --top-p        0.95 \
    --nmr-gen-topk 10 \
    --max-concurrency 8
```

Progress is checkpointed to `<output>.checkpoint.jsonl` the same way as
`python -m spectune.rollout` (pass `--no-checkpoint` to disable). Pass
`--no-hit-at-k` to skip the metrics report and only write the output file.
