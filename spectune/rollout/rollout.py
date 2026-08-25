"""Parallel offline LLM rollout with tool execution and rejection sampling."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectune.format.v1 import (
    TOOL_RESPONSE_END,
    TOOL_RESPONSE_START,
    extract_tool_calls,
    format_tools_block,
)
from spectune.jsonl import write_jsonl as _write_jsonl
from spectune.llm import LlmClient, LlmClientProtocol, LlmConfig
from spectune.tools import (
    DEFAULT_RL_TOOL_NAMES,
    ToolManager,
    ToolManagerConfig,
    compact_tool_payload,
    resolve_tool_names,
    schemas_for_names,
)

from .config import RolloutConfig
from .sampling.base import BaseSampler

JsonDict = dict[str, Any]


def _load_skill_texts(paths: tuple[str, ...]) -> list[str]:
    """Read skill file contents (default: ``.md``) in order, stripped of
    surrounding whitespace."""
    return [Path(p).expanduser().read_text(encoding="utf-8").strip() for p in paths]


@dataclass
class RolloutRecord:
    """One accepted (prompt, response) pair produced by the rollout pipeline."""

    sample_id: str
    gt_smiles: str
    # Full conversation ready for SFT: [system, user, ..., assistant]
    messages: list[JsonDict]
    reward_score: float
    reward_details: JsonDict
    n_rounds: int
    # Parallel to ``messages``; non-None only for assistant turns that returned
    # chain-of-thought text.  Omitted from to_dict() when all entries are empty.
    reasoning_content: list[str | None] | None = None

    def to_dict(self) -> JsonDict:
        d: JsonDict = {
            "sample_id": self.sample_id,
            "gt_smiles": self.gt_smiles,
            "messages": self.messages,
            "reward_score": self.reward_score,
            "reward_details": self.reward_details,
            "n_rounds": self.n_rounds,
        }
        if self.reasoning_content and any(self.reasoning_content):
            d["reasoning_content"] = self.reasoning_content
        return d


def compute_hit_at_k_metrics(
    records: list[RolloutRecord],
    *,
    total_samples: int | None = None,
) -> JsonDict:
    """Aggregate canonical ground-truth rank into ``hit@1`` through ``hit@all``.

    ``total_samples`` should include failed rollouts so those samples count as
    misses. When omitted, only completed records form the denominator.
    """
    denominator = len(records) if total_samples is None else total_samples
    if denominator < len(records):
        raise ValueError("total_samples cannot be smaller than the number of completed records")

    ranks: list[int | None] = []
    max_candidates = 0
    for record in records:
        details = record.reward_details
        rank = details.get("gt_rank")
        ranks.append(rank if isinstance(rank, int) and rank > 0 else None)
        candidates = details.get("canonical_candidates")
        if isinstance(candidates, list):
            max_candidates = max(max_candidates, len(candidates))

    max_rank = max((rank for rank in ranks if rank is not None), default=0)
    max_k = max(1, max_candidates, max_rank)
    divisor = float(denominator) if denominator else 1.0
    metrics: JsonDict = {
        "num_samples": denominator,
        "num_completed": len(records),
        "max_k": max_k,
    }
    for k in range(1, max_k + 1):
        metrics[f"hit@{k}"] = sum(rank is not None and rank <= k for rank in ranks) / divisor
    metrics["hit@all"] = sum(rank is not None for rank in ranks) / divisor
    return metrics


class Rollout:
    """Parallel offline LLM rollout with optional rejection sampling.

    For each input sample the pipeline:

    1. Builds the prompt from the sample's ``turns`` + configured system prompt
       (with the hermes tools block injected, like verl GRPO).
    2. Runs a tool-agent loop: the LLM generates, any hermes ``<tool_call>``
       blocks are executed via :class:`ToolManager` (honoring
       ``config.nmr_gen_topk`` for ``nmr_generate``), the results are fed back
       as ``<tool_response>`` turns, and generation repeats until the model
       stops calling tools or ``max_assistant_turns`` is exhausted.
    3. When ``config.max_rounds`` is ``None`` (or ``sampler`` is ``None``):
       the trajectory is accepted unconditionally.
    4. When ``max_rounds`` is set: the whole agent loop is retried up to that
       many times and returns on the first trajectory accepted by ``sampler``.
       If no round passes, the last successful response is kept as a fallback —
       no sample is silently dropped unless every LLM call for it fails.

    Multi-turn samples (n user turns) run one agent loop per user turn,
    threading the previous turns into the context before the next user turn is
    sent.  Only the final assistant response is evaluated.

    All samples are dispatched concurrently; per-sample retries are sequential.
    Concurrency is bounded by ``LlmConfig.max_concurrency``.
    """

    def __init__(
        self,
        config: RolloutConfig | None = None,
        llm_config: LlmConfig | None = None,
        *,
        llm: LlmClientProtocol | None = None,
    ) -> None:
        """``llm``, if given, is used as-is -- pass a pre-built
        :class:`~spectune.llm.litellm.LitellmClient` (or any other object
        satisfying :class:`~spectune.llm.base.LlmClientProtocol`) to swap
        backends without editing this module. Its config is assumed to
        already carry the desired sampling params; ``config``'s
        ``temperature``/``top_p``/``max_tokens``/``max_concurrency`` are only
        applied to the default :class:`LlmClient` path.
        """
        self.config = config or RolloutConfig()
        self.llm = llm or LlmClient(
            dataclasses.replace(
                llm_config or LlmConfig(),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                max_concurrency=self.config.max_concurrency,
            )
        )
        # One manager per run; nmr_gen_topk lands in NmrGenerateConfig so the
        # schema default and execution both honor it.
        self.tool_manager = ToolManager.from_config(ToolManagerConfig(nmr_gen_topk=self.config.nmr_gen_topk))
        names = tuple(self.config.tool_names) or DEFAULT_RL_TOOL_NAMES
        self.tool_names = resolve_tool_names(names, manager=self.tool_manager)
        schemas = schemas_for_names(self.tool_names, manager=self.tool_manager)
        system_prompt = self.config.system_prompt + "\n\n" + format_tools_block(schemas)
        for skill_text in _load_skill_texts(self.config.skills):
            system_prompt += "\n" + skill_text
        self.system_prompt = system_prompt

    async def _run_agent_loop(
        self, messages: list[JsonDict], reasoning_content: list[str | None], sample_id: str = ""
    ) -> str | None:
        """Generate → execute tool calls → feed results back, until answer or cap.

        Mutates ``messages`` and ``reasoning_content`` in place (both stay
        parallel: one entry per appended message).  Returns the final assistant
        response, or ``None`` if an LLM call fails outright.
        """
        final_response = ""
        complete_fn = getattr(self.llm, "complete_messages_with_reasoning", None)
        for turn_idx in range(self.config.max_assistant_turns):
            if complete_fn is not None:
                response, rc = await complete_fn(messages)
            else:
                response, rc = await self.llm.complete_messages(messages), ""
            if not response:
                return None
            messages.append({"role": "assistant", "content": response})
            reasoning_content.append(rc or None)
            final_response = response
            tool_calls = extract_tool_calls(response)
            if not tool_calls:
                if self.config.show_progress:
                    print(f"[agent] id={sample_id}  turn={turn_idx + 1}  done (no tool call)", flush=True)
                break
            if self.config.show_progress:
                names = ",".join(c["name"] for c in tool_calls)
                print(f"[agent] id={sample_id}  turn={turn_idx + 1}  tools={names}", flush=True)
            results = await asyncio.gather(
                *(self.tool_manager.invoke(call["name"], call["arguments"]) for call in tool_calls)
            )
            for result in results:
                payload = json.dumps(compact_tool_payload(result), ensure_ascii=False)
                messages.append({"role": "user", "content": f"{TOOL_RESPONSE_START}\n{payload}\n{TOOL_RESPONSE_END}"})
                reasoning_content.append(None)
        return final_response

    async def _sample_response(self, sample: JsonDict) -> tuple[str, list[JsonDict], list[str | None]] | None:
        """Draw one agent-loop trajectory for ``sample``.

        Returns ``(final_response, full_messages, reasoning_content)`` or ``None``
        on LLM failure.  ``reasoning_content`` is parallel to ``full_messages``
        (one entry per message; non-None only for assistant turns that returned
        chain-of-thought text).

        Handles n user turns: each user turn triggers a full agent loop, and the
        accumulated conversation is carried into the next turn.
        """
        turns = sample.get("turns") or []
        user_turns = [t for t in turns if isinstance(t, dict) and t.get("role") == "user"]
        if not user_turns:
            return None

        sample_id = str(sample.get("sample_id", "?"))
        messages: list[JsonDict] = [
            {"role": "system", "content": self.system_prompt},
        ]
        reasoning_content: list[str | None] = [None]  # system message has no reasoning
        final_response = ""
        for user_turn in user_turns:
            messages.append({"role": "user", "content": str(user_turn.get("content", ""))})
            reasoning_content.append(None)
            response = await self._run_agent_loop(messages, reasoning_content, sample_id=sample_id)
            if response is None:
                return None
            final_response = response

        return final_response, messages, reasoning_content

    async def _run_one(
        self,
        sample: JsonDict,
        sampler: BaseSampler | None,
        evaluator: Any,
    ) -> RolloutRecord | None:
        def _make_record(
            response: str,
            full_messages: list[JsonDict],
            reasoning_content: list[str | None],
            n_rounds: int,
        ) -> RolloutRecord:
            reward = evaluator.evaluate(response, sample.get("gt_smiles", ""))
            return RolloutRecord(
                sample_id=str(sample.get("sample_id", "")),
                gt_smiles=str(sample.get("gt_smiles", "")),
                messages=full_messages,
                reward_score=reward.score,
                reward_details=dict(reward.details),
                n_rounds=n_rounds,
                reasoning_content=reasoning_content,
            )

        # No rejection sampling: sample once and accept unconditionally.
        if sampler is None or self.config.max_rounds is None:
            result = await self._sample_response(sample)
            if result is None:
                return None
            response, full_messages, reasoning_content = result
            return _make_record(response, full_messages, reasoning_content, 1)

        # Rejection sampling: try up to max_rounds; fall back to the last
        # successful response if no round passes the sampler.
        last: tuple[str, list[JsonDict], list[str | None], int] | None = None
        for round_idx in range(1, self.config.max_rounds + 1):
            result = await self._sample_response(sample)
            if result is None:
                continue
            response, full_messages, reasoning_content = result
            last = (response, full_messages, reasoning_content, round_idx)
            if sampler.accept(response, sample):
                return _make_record(response, full_messages, reasoning_content, round_idx)

        if last is None:
            return None  # every LLM call failed
        return _make_record(*last)

    async def run(
        self,
        samples: list[JsonDict],
        sampler: BaseSampler | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[RolloutRecord]:
        """Run rollout for all ``samples`` concurrently.

        When ``sampler`` is ``None`` (or ``config.max_rounds`` is ``None``),
        each sample is called once and accepted unconditionally.  Otherwise
        rejection sampling is applied up to ``config.max_rounds`` times, with
        the last successful response kept as a fallback.

        When ``checkpoint_path`` is set, each accepted record is appended to
        that file immediately after completion so progress survives a crash.
        """
        from spectune.reward import RewardEvaluator

        evaluator = RewardEvaluator()
        total = len(samples)
        counter = [0]
        sample_sem = asyncio.Semaphore(self.config.max_concurrency)

        ckpt_fh = None
        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt_fh = ckpt.open("a", encoding="utf-8")

        async def _run_and_report(s: JsonDict) -> RolloutRecord | None:
            async with sample_sem:
                record = await self._run_one(s, sampler, evaluator)
            counter[0] += 1
            if self.config.show_progress:
                if record is not None:
                    print(
                        f"[rollout] {counter[0]}/{total}"
                        f"  id={record.sample_id}"
                        f"  rounds={record.n_rounds}"
                        f"  score={record.reward_score:.4f}",
                        flush=True,
                    )
                    print(record.messages)
                else:
                    last_error = getattr(self.llm, "last_error", None)
                    suffix = f"  error={last_error}" if last_error else ""
                    print(
                        f"[rollout] {counter[0]}/{total}"
                        f"  id={s.get('sample_id', '?')}  FAILED{suffix}",
                        flush=True,
                    )
            if record is not None and ckpt_fh is not None:
                ckpt_fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                ckpt_fh.flush()
            return record

        try:
            tasks = [_run_and_report(s) for s in samples]
            results = await asyncio.gather(*tasks)
        finally:
            if ckpt_fh is not None:
                ckpt_fh.close()

        return [r for r in results if r is not None]


def write_jsonl(records: list[RolloutRecord], path: str | Path) -> None:
    _write_jsonl(path, (record.to_dict() for record in records))


def write_parquet(records: list[RolloutRecord], path: str | Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("writing parquet requires pandas/pyarrow; install with `pip install spectune[data]`") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_dict() for r in records]).to_parquet(destination, index=False)


__all__ = ["Rollout", "RolloutRecord", "compute_hit_at_k_metrics", "write_jsonl", "write_parquet"]
