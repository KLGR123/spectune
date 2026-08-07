# Spectune reward

`spectune.reward` provides dependency-light reward shaping for molecular
structure rollouts. It accepts either OpenAI-style message dictionaries or a
decoded multi-turn response string.

## Format contract (`v1`)

Training uses a single answer / tool contract from `spectune.format.v1`:

```text
<tool_call>
{"name": "nmr_rerank", "arguments": {"smiles_list": ["CCO"]}}
</tool_call>
```

```json
{"smiles": ["CCO", "COC"]}
```

`RewardConfig.strict_answer_format=True` (default) only scores the JSON answer
above. Set it to `False` only for offline legacy audits. OpenAI-style structured
`message["tool_calls"]` remain accepted alongside hermes tags.

## Components

Four named raw components are computed first:

1. `gt_smiles`: a GT hit at rank `r` receives
   `gt_match_reward * rank_discount ** (r - 1)`.
2. `tool_call_format`: every malformed call, unknown tool, invalid argument
   JSON, or argument object that violates the tool schema receives
   `invalid_tool_call_penalty`.
3. `smiles_validity`: the rollout receives `invalid_smiles_penalty` once if
   any explicitly returned candidate cannot be parsed by RDKit.
4. `tool_call_count`: calls beyond `max_tool_calls` (default `8`) receive
   `excess_tool_call_penalty * excess_count`. `None` disables this component.

The final score is the weighted sum
`sum(component_weights[k] * components[k])`. Weights must be non-negative and
sum to `1`. Defaults favour GT learning:

| component | weight |
|---|---|
| `gt_smiles` | `0.7` |
| `tool_call_format` | `0.1` |
| `smiles_validity` | `0.1` |
| `tool_call_count` | `0.1` |

Raw term defaults are `1.0`, `0.8`, `-1.0`, `-1.0`, `max_tool_calls=8`, and
`-0.1` respectively.

## Python API

```python
from spectune import RewardConfig, RewardEvaluator, ToolManager
from spectune.format.v1 import format_final_answer

manager = ToolManager.from_config()
reward = RewardEvaluator.from_tool_manager(
    manager,
    RewardConfig(
        max_tool_calls=3,
        rank_discount=0.7,
        component_weights={
            "gt_smiles": 0.7,
            "tool_call_format": 0.1,
            "smiles_validity": 0.1,
            "tool_call_count": 0.1,
        },
    ),
)
result = reward.evaluate(format_final_answer(["CCO"]), {"gt_smiles": "CCO"})

print(result.score)
print(result.components)
print(result.details)
```

Install `spectune[chem]` in training environments. Without RDKit, values are
compared as opaque strings and invalid-SMILES detection is disabled.

## verl handoff

For a scalar custom reward, configure verl with a real file path (stock verl
does not accept `pkg://` here):

```text
custom_reward_function.path=/abs/path/to/spectune/reward/verl.py
custom_reward_function.name=compute_score
```

It accepts verl's `data_source`, `solution_str`, `ground_truth`, and `extra_info`
arguments. Per-sample settings can be supplied as:

```python
extra_info = {
    "tool_names": ["nmr_rerank", "web_search"],
    "reward_config": {
        "max_tool_calls": 3,
        "rank_discount": 0.7,
        "excess_tool_call_penalty": -0.2,
        "component_weights": {
            "gt_smiles": 0.7,
            "tool_call_format": 0.1,
            "smiles_validity": 0.1,
            "tool_call_count": 0.1,
        },
    }
}
```

When available, put structured messages in `extra_info["rollout_messages"]` and
the exact rollout schemas in `extra_info["tool_schemas"]`. Otherwise the adapter:

1. rebuilds assistant / tool turns from the decoded response string, isolating
   the final `{"smiles":[...]}` answer from glued tool payloads;
2. resolves schemas from `tool_schemas`, else `tool_names`, else `tools_kwargs`
   keys, else `DEFAULT_RL_TOOL_NAMES`.

Batch reward managers can import `spectune.reward.verl.compute_score_batched`.

See `examples/verl/README.md` for dataset compilation and tool YAML wiring.
