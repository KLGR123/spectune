import pytest

from spectune import RewardConfig, RewardEvaluator
from spectune.format.v1 import format_final_answer
from spectune.reward.verl import compute_score, compute_score_batched

_DEFAULT_GT_WEIGHT = 0.7
_DEFAULT_VALIDITY_WEIGHT = 0.1


def _tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "rank_structures",
            "description": "Rank candidate structures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles_list": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "topk": {"type": "integer", "minimum": 1},
                },
                "required": ["smiles_list"],
                "additionalProperties": False,
            },
        },
    }


def test_gt_reward_uses_canonical_smiles_rank_and_discount():
    evaluator = RewardEvaluator(RewardConfig(rank_discount=0.5))

    result = evaluator.evaluate(format_final_answer(["CCC", "OCC"]), "CCO")

    assert result.components["gt_smiles"] == pytest.approx(0.5)
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT * 0.5)
    assert result.details["gt_rank"] == 2


def test_free_form_literal_gt_requires_legacy_mode():
    strict = RewardEvaluator().evaluate("The final structure is CCO.", "CCO")
    assert strict.details["gt_rank"] is None

    legacy = RewardEvaluator(RewardConfig(strict_answer_format=False)).evaluate(
        "The final structure is CCO.",
        "CCO",
    )
    assert legacy.components["gt_smiles"] == 1.0
    assert legacy.details["gt_rank"] == 1
    assert legacy.score == pytest.approx(_DEFAULT_GT_WEIGHT)


def test_invalid_smiles_is_penalized_once():
    result = RewardEvaluator().evaluate(format_final_answer(["CCO", "C1CC", "C1CCC"]), "CCO")

    assert result.components["smiles_validity"] == -1.0
    assert result.details["invalid_smiles"] == ["C1CC", "C1CCC"]
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT * 1.0 + _DEFAULT_VALIDITY_WEIGHT * -1.0)


def test_structured_tool_calls_are_validated_against_schema():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "rank_structures",
                        "arguments": '{"smiles_list": ["CCO"], "topk": 1}',
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "rank_structures",
                        "arguments": {"topk": 0, "unexpected": True},
                    },
                },
            ],
        },
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": format_final_answer(["CCO"])},
    ]
    evaluator = RewardEvaluator(tool_schemas=[_tool_schema()])

    result = evaluator.evaluate(messages, "CCO")

    assert result.components["tool_call_format"] == -1.0
    assert result.details["invalid_tool_call_count"] == 1
    assert len(result.details["invalid_tool_calls"]) >= 2


def test_decoded_tool_calls_and_excess_count_are_penalized():
    rollout = f"""
<tool_call>
{{"name": "rank_structures", "arguments": {{"smiles_list": ["CCO"]}}}}
</tool_call>
<tool_call>
{{"name": "missing_tool", "arguments": {{}}}}
</tool_call>
<tool_call>
not-json
</tool_call>

{format_final_answer(["CCC", "CCO"])}
"""
    config = RewardConfig(
        rank_discount=0.5,
        max_tool_calls=1,
        excess_tool_call_penalty=-0.2,
        component_weights={
            "gt_smiles": 0.25,
            "tool_call_format": 0.25,
            "smiles_validity": 0.25,
            "tool_call_count": 0.25,
        },
    )
    result = RewardEvaluator(config, tool_schemas=[_tool_schema()]).evaluate(rollout, "CCO")

    assert result.components == {
        "gt_smiles": pytest.approx(0.5),
        "tool_call_format": pytest.approx(-2.0),
        "smiles_validity": 0.0,
        "tool_call_count": pytest.approx(-0.4),
    }
    assert result.details["tool_call_count"] == 3
    assert result.score == pytest.approx(0.25 * -1.9)


def test_component_weights_scale_final_score():
    config = RewardConfig(
        component_weights={
            "gt_smiles": 0.7,
            "tool_call_format": 0.1,
            "smiles_validity": 0.1,
            "tool_call_count": 0.1,
        }
    )
    result = RewardEvaluator(config).evaluate(format_final_answer(["CCO", "C1CC"]), "CCO")

    assert result.components["gt_smiles"] == 1.0
    assert result.components["smiles_validity"] == -1.0
    assert result.score == pytest.approx(0.7 * 1.0 + 0.1 * -1.0)
    assert result.details["weighted_components"]["gt_smiles"] == pytest.approx(0.7)


def test_verl_scalar_and_batch_adapters():
    extra_info = {
        "tool_schemas": [_tool_schema()],
        "reward_config": {
            "rank_discount": 0.25,
            "max_tool_calls": 0,
            "component_weights": {
                "gt_smiles": 1.0,
                "tool_call_format": 0.0,
                "smiles_validity": 0.0,
                "tool_call_count": 0.0,
            },
        },
    }

    assert compute_score(format_final_answer(["CCC", "CCO"]), "CCO", extra_info=extra_info)["score"] == pytest.approx(
        0.25
    )
    batched = compute_score_batched(
        [format_final_answer(["CCO"]), format_final_answer(["CCC"])],
        ["CCO", "CCO"],
        extra_infos=[extra_info, extra_info],
    )
    assert [result["score"] for result in batched] == [pytest.approx(1.0), pytest.approx(0.0)]


def test_verl_adapter_resolves_schemas_from_tool_names():
    rollout = (
        '<tool_call>{"name": "rank_structures", "arguments": {"smiles_list": ["CCO"]}}</tool_call>\n'
        + format_final_answer(["CCO"])
    )
    score = compute_score(
        rollout,
        "CCO",
        extra_info={
            "tool_schemas": [_tool_schema()],
            "tool_names": ["rank_structures"],
            "reward_config": {
                "component_weights": {
                    "gt_smiles": 1.0,
                    "tool_call_format": 0.0,
                    "smiles_validity": 0.0,
                    "tool_call_count": 0.0,
                }
            },
        },
    )
    assert score["score"] == pytest.approx(1.0)
    assert score["gt_smiles"] == pytest.approx(1.0)


def test_reward_ignores_tool_result_smiles_in_decoded_rollout():
    """Tool responses glued into verl's decoded string must not steal the GT match."""
    tool_result = json_dumps_tool_result_with_smiles(["CCC"])
    rollout = (
        '<tool_call>{"name": "rank_structures", "arguments": {"smiles_list": ["CCO"]}}</tool_call>\n'
        f"{tool_result}\n"
        f"{format_final_answer(['CCO'])}"
    )
    result = RewardEvaluator(tool_schemas=[_tool_schema()]).evaluate(rollout, "CCO")
    assert result.details["answer_candidates"] == ["CCO"]
    assert result.details["gt_rank"] == 1


def json_dumps_tool_result_with_smiles(smiles_list: list[str]) -> str:
    import json

    return json.dumps(
        {
            "completion": "success",
            "status": "ok",
            "data": {"candidates": [{"smiles": s} for s in smiles_list]},
        },
        ensure_ascii=False,
    )


def test_default_max_tool_calls_matches_agent_budget():
    assert RewardConfig().max_tool_calls == 8


def test_reward_config_rejects_positive_penalties():
    with pytest.raises(ValueError, match="non-positive"):
        RewardConfig(invalid_smiles_penalty=1.0)


def test_reward_config_requires_component_weights_to_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        RewardConfig(
            component_weights={
                "gt_smiles": 0.5,
                "tool_call_format": 0.2,
                "smiles_validity": 0.1,
                "tool_call_count": 0.1,
            }
        )


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "match"),
    [
        ({"gt_match_reward": -0.1}, ValueError, "non-negative"),
        ({"rank_discount": -0.1}, ValueError, "between 0 and 1"),
        ({"rank_discount": 1.1}, ValueError, "between 0 and 1"),
        ({"invalid_tool_call_penalty": 0.1}, ValueError, "non-positive"),
        ({"invalid_smiles_penalty": 0.1}, ValueError, "non-positive"),
        ({"max_tool_calls": -1}, ValueError, "non-negative or None"),
        ({"excess_tool_call_penalty": 0.1}, ValueError, "non-positive"),
        ({"component_weights": ["not", "a", "mapping"]}, TypeError, "must be a mapping"),
        (
            {"component_weights": {"gt_smiles": 0.7, "tool_call_format": 0.1, "smiles_validity": 0.2}},
            ValueError,
            "missing",
        ),
        (
            {
                "component_weights": {
                    "gt_smiles": 0.7,
                    "tool_call_format": 0.1,
                    "smiles_validity": 0.1,
                    "tool_call_count": 0.05,
                    "unexpected": 0.05,
                }
            },
            ValueError,
            "unknown",
        ),
        (
            {
                "component_weights": {
                    "gt_smiles": -0.1,
                    "tool_call_format": 0.4,
                    "smiles_validity": 0.4,
                    "tool_call_count": 0.3,
                }
            },
            ValueError,
            "non-negative",
        ),
    ],
)
def test_reward_config_rejects_invalid_values(kwargs, exc_type, match):
    with pytest.raises(exc_type, match=match):
        RewardConfig(**kwargs)
