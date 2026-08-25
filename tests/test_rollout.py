import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from spectune.format.v1 import format_final_answer
from spectune.llm import LlmClientProtocol, LlmConfig
from spectune.reward import RewardEvaluator
from spectune.rollout import Rollout, RolloutConfig, RolloutRecord, compute_hit_at_k_metrics
from spectune.rollout.sampling.rejection import GtRejectionSampler
from spectune.tools.base import ToolResult
from spectune.tools.config import NMR_GENERATE_MAX_TOPK


def _rollout(**config_kwargs):
    config = RolloutConfig(show_progress=False, **config_kwargs)
    return Rollout(config, LlmConfig())


def _fake_llm_client() -> LlmClientProtocol:
    """Build a minimal fake client satisfying the protocol."""
    client = MagicMock(spec=LlmClientProtocol)
    client.stats = {"requests": 0, "failures": 0, "retries": 0}
    client.last_error = None
    client.available = True
    client.complete = AsyncMock(return_value="ok")
    client.complete_messages = AsyncMock(return_value="ok")
    client.complete_many = AsyncMock(return_value=["ok"])
    return client


class TestRolloutClientInjection:
    def test_uses_injected_client_directly(self):
        config = RolloutConfig(show_progress=False)
        fake = _fake_llm_client()
        rollout = Rollout(config, llm=fake)

        assert rollout.llm is fake
        assert rollout.llm.available

    def test_injected_client_bypasses_llm_config(self):
        config = RolloutConfig(temperature=0.1, top_p=0.8, max_tokens=128, show_progress=False)
        fake = _fake_llm_client()
        fake.config = MagicMock(temperature=0.9, top_p=0.95, max_tokens=512)

        rollout = Rollout(config, llm=fake)

        assert rollout.llm.config.temperature == 0.9
        assert rollout.llm.config.top_p == 0.95
        assert rollout.llm.config.max_tokens == 512


class TestRunAgentLoop:
    def test_stops_when_response_has_no_tool_call(self):
        rollout = _rollout(max_assistant_turns=5)
        rollout.llm.complete_messages = AsyncMock(return_value='{"smiles": ["CCO"]}')
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        reasoning_content: list = []

        response = asyncio.run(rollout._run_agent_loop(messages, reasoning_content, sample_id="s1"))

        assert response == '{"smiles": ["CCO"]}'
        assert messages[-1] == {"role": "assistant", "content": '{"smiles": ["CCO"]}'}
        rollout.llm.complete_messages.assert_awaited_once()

    def test_executes_tool_call_then_returns_final_answer(self):
        rollout = _rollout(max_assistant_turns=5)
        rollout.llm.complete_messages = AsyncMock(
            side_effect=[
                '<tool_call>\n{"name": "nmr_generate", "arguments": {}}\n</tool_call>',
                '{"smiles": ["CCO"]}',
            ]
        )
        rollout.tool_manager.invoke = AsyncMock(
            return_value=ToolResult(completion="success", status="ok", data={"candidates": []})
        )
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        reasoning_content: list = []

        response = asyncio.run(rollout._run_agent_loop(messages, reasoning_content, sample_id="s2"))

        assert response == '{"smiles": ["CCO"]}'
        assert rollout.llm.complete_messages.await_count == 2
        rollout.tool_manager.invoke.assert_awaited_once_with("nmr_generate", {})
        tool_response_messages = [m for m in messages if m["role"] == "user" and "tool_response" in m["content"]]
        assert len(tool_response_messages) == 1

    def test_stops_at_max_assistant_turns_when_tool_calls_never_end(self):
        rollout = _rollout(max_assistant_turns=2)
        rollout.llm.complete_messages = AsyncMock(
            return_value='<tool_call>\n{"name": "nmr_generate", "arguments": {}}\n</tool_call>'
        )
        rollout.tool_manager.invoke = AsyncMock(return_value=ToolResult(completion="success", status="ok", data={}))
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        reasoning_content: list = []

        response = asyncio.run(rollout._run_agent_loop(messages, reasoning_content, sample_id="s3"))

        assert rollout.llm.complete_messages.await_count == 2
        assert response == '<tool_call>\n{"name": "nmr_generate", "arguments": {}}\n</tool_call>'

    def test_returns_none_when_llm_call_fails(self):
        rollout = _rollout()
        rollout.llm.complete_messages = AsyncMock(return_value="")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        reasoning_content: list = []

        response = asyncio.run(rollout._run_agent_loop(messages, reasoning_content, sample_id="s4"))

        assert response is None


class TestRunOne:
    def test_accepts_first_response_when_no_sampler(self):
        rollout = _rollout()
        rollout._sample_response = AsyncMock(return_value=(format_final_answer(["CCO"]), [], []))
        sample = {"sample_id": "s1", "gt_smiles": "CCO"}

        record = asyncio.run(rollout._run_one(sample, sampler=None, evaluator=RewardEvaluator()))

        assert record.n_rounds == 1
        assert record.sample_id == "s1"
        rollout._sample_response.assert_awaited_once()

    def test_rejection_sampling_accepts_first_passing_round(self):
        rollout = _rollout(max_rounds=3)
        rollout._sample_response = AsyncMock(
            side_effect=[
                (format_final_answer(["CCN"]), ["m1"], []),
                (format_final_answer(["CCO"]), ["m2"], []),
            ]
        )
        sampler = GtRejectionSampler()
        sample = {"sample_id": "s2", "gt_smiles": "CCO"}

        record = asyncio.run(rollout._run_one(sample, sampler=sampler, evaluator=RewardEvaluator()))

        assert record.n_rounds == 2
        assert record.messages == ["m2"]
        assert rollout._sample_response.await_count == 2

    def test_rejection_sampling_falls_back_to_last_response_when_none_pass(self):
        rollout = _rollout(max_rounds=2)
        rollout._sample_response = AsyncMock(
            side_effect=[
                (format_final_answer(["CCN"]), ["m1"], []),
                (format_final_answer(["CCN"]), ["m2"], []),
            ]
        )
        sampler = GtRejectionSampler()
        sample = {"sample_id": "s3", "gt_smiles": "CCO"}

        record = asyncio.run(rollout._run_one(sample, sampler=sampler, evaluator=RewardEvaluator()))

        assert record is not None
        assert record.n_rounds == 2
        assert record.messages == ["m2"]

    def test_returns_none_when_every_sample_response_fails(self):
        rollout = _rollout(max_rounds=2)
        rollout._sample_response = AsyncMock(return_value=None)
        sampler = GtRejectionSampler()
        sample = {"sample_id": "s4", "gt_smiles": "CCO"}

        record = asyncio.run(rollout._run_one(sample, sampler=sampler, evaluator=RewardEvaluator()))

        assert record is None


class TestGtRejectionSampler:
    def test_accepts_when_gt_smiles_is_in_the_final_answer(self):
        sampler = GtRejectionSampler()
        response = format_final_answer(["CCC", "CCO"])

        assert sampler.accept(response, {"gt_smiles": "CCO"}) is True

    def test_rejects_when_gt_smiles_is_absent(self):
        sampler = GtRejectionSampler()
        response = format_final_answer(["CCC", "CCN"])

        assert sampler.accept(response, {"gt_smiles": "CCO"}) is False

    def test_rejects_when_sample_has_no_gt_smiles(self):
        sampler = GtRejectionSampler()
        response = format_final_answer(["CCO"])

        assert sampler.accept(response, {"gt_smiles": ""}) is False


class TestHitAtKMetrics:
    @staticmethod
    def _record(rank, candidates):
        return RolloutRecord(
            sample_id=str(rank),
            gt_smiles="CCO",
            messages=[],
            reward_score=0.0,
            reward_details={"gt_rank": rank, "canonical_candidates": candidates},
            n_rounds=1,
        )

    def test_reports_every_k_and_counts_failed_rollouts_as_misses(self):
        records = [
            self._record(1, ["CCO", "CCC", "CCN"]),
            self._record(3, ["CCC", "CCN", "CCO"]),
            self._record(None, ["CCC", "CCN"]),
        ]

        metrics = compute_hit_at_k_metrics(records, total_samples=4)

        assert metrics["num_samples"] == 4
        assert metrics["num_completed"] == 3
        assert metrics["max_k"] == 3
        assert metrics["hit@1"] == pytest.approx(0.25)
        assert metrics["hit@2"] == pytest.approx(0.25)
        assert metrics["hit@3"] == pytest.approx(0.5)
        assert metrics["hit@all"] == pytest.approx(0.5)

    def test_rejects_a_denominator_smaller_than_completed_records(self):
        with pytest.raises(ValueError, match="total_samples"):
            compute_hit_at_k_metrics([self._record(1, ["CCO"])], total_samples=0)


class TestRolloutConfigValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_rounds": 0}, "max_rounds"),
            ({"temperature": 0.0}, "temperature"),
            ({"temperature": 2.1}, "temperature"),
            ({"top_p": 0.0}, "top_p"),
            ({"top_p": 1.1}, "top_p"),
            ({"max_tokens": 0}, "max_tokens"),
            ({"nmr_gen_topk": 0}, "nmr_gen_topk"),
            ({"nmr_gen_topk": NMR_GENERATE_MAX_TOPK + 1}, "nmr_gen_topk"),
            ({"max_assistant_turns": 0}, "max_assistant_turns"),
            ({"max_concurrency": 0}, "max_concurrency"),
        ],
    )
    def test_rejects_invalid_values(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RolloutConfig(**kwargs)
