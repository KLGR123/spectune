import json

import pytest

from spectune import RewardConfig, RewardEvaluator
from spectune.format.v1 import format_final_answer
from spectune.reward.verl import compute_score, compute_score_batched

_DEFAULT_GT_WEIGHT = 0.7
_DEFAULT_VALIDITY_WEIGHT = -0.1


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

    assert result.components["gt_smiles"] == pytest.approx(_DEFAULT_GT_WEIGHT * 0.5)
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT * 0.5)
    assert result.details["gt_rank"] == 2


def test_free_form_literal_gt_requires_legacy_mode():
    strict = RewardEvaluator().evaluate("The final structure is CCO.", "CCO")
    assert strict.details["gt_rank"] is None

    legacy = RewardEvaluator(RewardConfig(strict_answer_format=False)).evaluate(
        "The final structure is CCO.",
        "CCO",
    )
    assert legacy.components["gt_smiles"] == pytest.approx(_DEFAULT_GT_WEIGHT)
    assert legacy.details["gt_rank"] == 1
    assert legacy.score == pytest.approx(_DEFAULT_GT_WEIGHT)


def test_empty_gt_rewarded_only_for_explicit_empty_answer():
    evaluator = RewardEvaluator()

    # Correct: agent explicitly outputs an empty list
    correct = evaluator.evaluate(format_final_answer([]), "")
    assert correct.components["gt_smiles"] == pytest.approx(_DEFAULT_GT_WEIGHT)
    assert correct.details["gt_rank"] is None

    # Wrong: agent outputs a SMILES when gt is empty
    wrong = evaluator.evaluate(format_final_answer(["CCO"]), "")
    assert wrong.components["gt_smiles"] == pytest.approx(0.0)

    # Wrong: agent outputs unparseable text (empty result not from {"smiles": []})
    garbage = evaluator.evaluate([{"role": "assistant", "content": "No structure found."}], "")
    assert garbage.components["gt_smiles"] == pytest.approx(0.0)


def test_invalid_smiles_is_penalized_once():
    result = RewardEvaluator().evaluate(format_final_answer(["CCO", "C1CC", "C1CCC"]), "CCO")

    assert result.components["smiles_validity"] == pytest.approx(_DEFAULT_VALIDITY_WEIGHT)
    assert result.details["invalid_smiles"] == ["C1CC", "C1CCC"]
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT + _DEFAULT_VALIDITY_WEIGHT)


def test_loose_match_scores_lower_than_exact_match_when_exact_misses():
    # D-alanine vs L-alanine: same skeleton, opposite chirality at the one
    # stereocenter -- exact (isomeric) match misses, lenient match hits rank 1.
    result = RewardEvaluator().evaluate(format_final_answer(["C[C@@H](N)C(=O)O"]), "C[C@H](N)C(=O)O")

    assert result.components["gt_smiles"] == pytest.approx(0.0)
    assert result.details["gt_rank"] is None
    assert result.details["gt_loose_rank"] == 1
    assert result.details["gt_loose_match_bonus"] == pytest.approx(0.35)
    assert result.score == pytest.approx(0.35)


def test_exact_match_takes_priority_over_loose_match():
    result = RewardEvaluator().evaluate(format_final_answer(["CCO"]), "CCO")

    assert result.details["gt_rank"] == 1
    assert result.details["gt_loose_rank"] is None
    assert result.details["gt_loose_match_bonus"] == pytest.approx(0.0)
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT * 1.0)


def test_loose_match_bonus_is_discounted_by_rank():
    result = RewardEvaluator(RewardConfig(rank_discount=0.5)).evaluate(
        format_final_answer(["CCC", "CCN", "C[C@@H](N)C(=O)O"]),
        "C[C@H](N)C(=O)O",
    )

    assert result.details["gt_rank"] is None
    assert result.details["gt_loose_rank"] == 3
    assert result.details["gt_loose_match_bonus"] == pytest.approx(0.35 * 0.5**2)
    assert result.score == pytest.approx(0.35 * 0.5**2)


def test_loose_match_reward_is_configurable():
    result = RewardEvaluator(RewardConfig(gt_loose_match_reward=0.5)).evaluate(
        format_final_answer(["C[C@@H](N)C(=O)O"]),
        "C[C@H](N)C(=O)O",
    )

    assert result.details["gt_loose_match_bonus"] == pytest.approx(0.5)
    assert result.score == pytest.approx(0.5)


def test_loose_match_does_not_conflate_diastereomers():
    # 2,3-dichlorobutane: partial inversion at one of two centers is a true
    # diastereomer, not a mirror image -- must NOT be treated as a lenient hit.
    result = RewardEvaluator().evaluate(
        format_final_answer(["C[C@H](Cl)[C@@H](Cl)C"]),
        "C[C@H](Cl)[C@H](Cl)C",
    )

    assert result.details["gt_rank"] is None
    assert result.details["gt_loose_rank"] is None
    assert result.score == pytest.approx(0.0)


def test_gt_loose_match_reward_config_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        RewardConfig(gt_loose_match_reward=-0.01)


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

    assert result.components["tool_call_format"] == pytest.approx(-0.1)
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
    )
    result = RewardEvaluator(config, tool_schemas=[_tool_schema()]).evaluate(rollout, "CCO")

    assert result.components == {
        "gt_smiles": pytest.approx(_DEFAULT_GT_WEIGHT * 0.5),
        "tool_call_format": pytest.approx(-0.2),
        "smiles_validity": 0.0,
        "tool_call_count": pytest.approx(-0.4),
    }
    assert result.details["tool_call_count"] == 3
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT * 0.5 - 0.2 - 0.4)


def test_components_sum_directly_to_final_score():
    result = RewardEvaluator().evaluate(format_final_answer(["CCO", "C1CC"]), "CCO")

    assert result.components["gt_smiles"] == pytest.approx(_DEFAULT_GT_WEIGHT)
    assert result.components["smiles_validity"] == pytest.approx(_DEFAULT_VALIDITY_WEIGHT)
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT + _DEFAULT_VALIDITY_WEIGHT)
    assert result.details["components"] == result.components


def test_verl_scalar_and_batch_adapters():
    extra_info = {
        "tool_schemas": [_tool_schema()],
        "reward_config": {
            "rank_discount": 0.25,
            "max_tool_calls": 0,
            "gt_match_reward": 1.0,
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
            "reward_config": {"gt_match_reward": 1.0},
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


def _code_interpreter_call(code: str) -> dict:
    return {
        "type": "function",
        "function": {"name": "code_interpreter", "arguments": json.dumps({"code": code})},
    }


def _tool_response(status: str, stdout: str = "") -> dict:
    return {
        "role": "tool",
        "content": json.dumps({"completion": "success" if status == "ok" else "failure", "status": status, "data": {"stdout": stdout}}),
    }


def test_code_interpreter_missing_print_is_penalized_per_call():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_code_interpreter_call("x = 1 + 1")]},
        _tool_response("ok"),
        {"role": "assistant", "content": "", "tool_calls": [_code_interpreter_call("print(1 + 1)")]},
        _tool_response("ok", stdout="2"),
        {"role": "assistant", "content": format_final_answer(["CCO"])},
    ]

    result = RewardEvaluator().evaluate(messages, "CCO")

    assert result.details["code_interpreter_call_count"] == 2
    assert result.details["code_interpreter_missing_print_count"] == 1
    assert result.details["code_interpreter_missing_print_penalty"] == pytest.approx(-0.02)
    assert result.details["code_interpreter_error_count"] == 0
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT * 1.0 - 0.02)


def test_code_interpreter_error_is_penalized_per_call():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [_code_interpreter_call("print(Chem.MolFromBadApi())")]},
        _tool_response("error"),
        {"role": "assistant", "content": "", "tool_calls": [_code_interpreter_call("print(rdkit_call())")]},
        _tool_response("error"),
        {"role": "assistant", "content": format_final_answer(["CCO"])},
    ]

    result = RewardEvaluator().evaluate(messages, "CCO")

    assert result.details["code_interpreter_call_count"] == 2
    assert result.details["code_interpreter_missing_print_count"] == 0
    assert result.details["code_interpreter_error_count"] == 2
    assert result.details["code_interpreter_error_penalty"] == pytest.approx(-0.1)
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT * 1.0 - 0.1)


def test_code_interpreter_penalties_accumulate_and_are_configurable():
    rollout = (
        '<tool_call>{"name": "code_interpreter", "arguments": {"code": "x = 1"}}</tool_call>\n'
        + json.dumps({"completion": "failure", "status": "error", "data": {}})
        + "\n"
        + '<tool_call>{"name": "code_interpreter", "arguments": {"code": "y = 2"}}</tool_call>\n'
        + json.dumps({"completion": "failure", "status": "error", "data": {}})
        + "\n"
        + format_final_answer(["CCO"])
    )
    config = RewardConfig(code_interpreter_missing_print_penalty=-0.03, code_interpreter_error_penalty=-0.07)

    result = RewardEvaluator(config).evaluate(rollout, "CCO")

    # Two calls, each missing print and erroring: penalties sum across the trajectory.
    assert result.details["code_interpreter_missing_print_penalty"] == pytest.approx(-0.06)
    assert result.details["code_interpreter_error_penalty"] == pytest.approx(-0.14)
    assert result.score == pytest.approx(_DEFAULT_GT_WEIGHT * 1.0 - 0.06 - 0.14)


def test_code_interpreter_penalty_config_rejects_positive_values():
    with pytest.raises(ValueError, match="non-positive"):
        RewardConfig(code_interpreter_missing_print_penalty=0.01)
    with pytest.raises(ValueError, match="non-positive"):
        RewardConfig(code_interpreter_error_penalty=0.01)


def test_default_max_tool_calls_matches_agent_budget():
    from spectune.rollout.config import DEFAULT_MAX_ASSISTANT_TURNS

    assert RewardConfig().max_tool_calls == DEFAULT_MAX_ASSISTANT_TURNS


def test_reward_config_rejects_positive_penalties():
    with pytest.raises(ValueError, match="non-positive"):
        RewardConfig(invalid_smiles_penalty=1.0)


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
    ],
)
def test_reward_config_rejects_invalid_values(kwargs, exc_type, match):
    with pytest.raises(exc_type, match=match):
        RewardConfig(**kwargs)
