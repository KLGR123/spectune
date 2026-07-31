"""Preferred answer and tool formats for Spectune rewards."""

import json

from spectune import RewardConfig, RewardEvaluator
from spectune.format import (
    extract_smiles_candidates,
    format_final_answer,
    messages_from_decoded_hermes,
)


def test_format_v1_extracts_ranked_smiles():
    assert extract_smiles_candidates(format_final_answer(["CCO", "COC"])) == ["CCO", "COC"]
    assert extract_smiles_candidates('prefix\n{"smiles": ["CCC"]}\n') == ["CCC"]
    assert extract_smiles_candidates(
        '<tool_call>{"name": "nmr_rerank", "arguments": {}}</tool_call>\n{"smiles": ["CCO"]}'
    ) == ["CCO"]


def test_final_answer_ignores_json_before_last_tool_call():
    rollout = (
        '{"smiles": ["CCC"]}\n'
        '<tool_call>{"name": "nmr_rerank", "arguments": {"smiles_list": ["CCC"]}}</tool_call>\n'
        '{"smiles": ["CCO"]}'
    )
    assert extract_smiles_candidates(rollout) == ["CCO"]
    result = RewardEvaluator().evaluate(rollout, "CCO")
    assert result.details["gt_rank"] == 1
    assert result.details["answer_candidates"] == ["CCO"]


def test_messages_from_decoded_hermes_splits_tool_response_and_answer():
    tool_payload = json.dumps(
        {"completion": "success", "status": "ok", "data": {"candidates": [{"smiles": "CCC"}]}},
        ensure_ascii=False,
    )
    text = (
        '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>\n'
        f"{tool_payload}\n"
        '{"smiles": ["CCO"]}'
    )
    messages = messages_from_decoded_hermes(text)
    assert messages[0]["role"] == "assistant"
    assert "<tool_call>" in messages[0]["content"]
    assert any(message["role"] == "tool" for message in messages)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == '{"smiles": ["CCO"]}'
    assert extract_smiles_candidates(text) == ["CCO"]


def test_strict_reward_requires_v1_json():
    hit = RewardEvaluator().evaluate(format_final_answer(["CCC", "CCO"]), "CCO")
    assert hit.details["gt_rank"] == 2

    miss = RewardEvaluator().evaluate("The final structure is CCO.", "CCO")
    assert miss.details["gt_rank"] is None
    assert miss.components["gt_smiles"] == 0.0


def test_legacy_reward_still_accepts_free_form():
    result = RewardEvaluator(RewardConfig(strict_answer_format=False)).evaluate(
        "The final structure is CCO.",
        "CCO",
    )
    assert result.details["gt_rank"] == 1
