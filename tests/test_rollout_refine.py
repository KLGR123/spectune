import asyncio
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from spectune.format.v1 import SYSTEM_PROMPT_THINK_RDKIT  # noqa: E402
from spectune.rollout.refine import (  # noqa: E402
    TrajectoryRefiner,
    build_parser,
    parse_assistant_rewrites,
    refine_parquet,
)


def _system_prompt() -> str:
    return """old prompt

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"nmr_generate","parameters":{"type":"object"}}}
</tools>

For each function call, return JSON within tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def _tool_response(value: int = 1) -> str:
    return f'<tool_response>\n{{"status":"ok","data":{{"value":{value}}}}}\n</tool_response>'


def _source_messages() -> list[dict]:
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": "identify this spectrum"},
        {
            "role": "assistant",
            "content": '<tool_call>\n{"name":"nmr_generate","arguments":{}}\n</tool_call>',
        },
        {"role": "user", "content": _tool_response()},
        {"role": "assistant", "content": '{"smiles":["CCO"]}'},
    ]


def _assistant_rewrites(*, second_reasoning: str = "工具证据支持乙醇，给出最终候选。") -> list[dict]:
    return [
        {
            "message_index": 2,
            "reasoning": "先核对分子式和谱峰约束，再调用已有生成工具区分候选。",
        },
        {
            "message_index": 4,
            "reasoning": second_reasoning,
        },
    ]


class FakeLlm:
    def __init__(self, response: str):
        self.response = response
        self.available = True
        self.last_error = None
        self.stats = {"requests": 0, "failures": 0, "retries": 0}
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def _response(rewrites: list[dict] | None = None, *, fenced: bool = False) -> str:
    payload = json.dumps(
        {"assistant_rewrites": rewrites if rewrites is not None else _assistant_rewrites()},
        ensure_ascii=False,
    )
    return f"```json\n{payload}\n```" if fenced else payload


def test_parse_assistant_rewrites_accepts_fenced_json():
    parsed = parse_assistant_rewrites(_response(fenced=True))

    assert parsed == {
        2: "先核对分子式和谱峰约束，再调用已有生成工具区分候选。",
        4: "工具证据支持乙醇，给出最终候选。",
    }


def test_parse_assistant_rewrites_rejects_format_tags():
    rewrites = _assistant_rewrites()
    rewrites[0]["reasoning"] = "<think>不要自行输出标签</think>"

    with pytest.raises(ValueError, match="forbidden format tag"):
        parse_assistant_rewrites(_response(rewrites))


def test_refiner_preserves_evidence_and_normalizes_system_prompt():
    llm = FakeLlm(_response())
    refiner = TrajectoryRefiner(llm, model_name="teacher")

    refined = asyncio.run(refiner.refine(_source_messages()))

    assert refined[0]["content"].startswith(SYSTEM_PROMPT_THINK_RDKIT.rstrip())
    assert "# Tools" in refined[0]["content"]
    assert refined[1]["content"] == "identify this spectrum"
    assert refined[3]["content"] == _tool_response()
    assert refined[2]["content"].startswith("<think>\n先核对分子式")
    assert refined[2]["content"].endswith(
        '<tool_call>\n{"name": "nmr_generate", "arguments": {}}\n</tool_call>'
    )
    assert refined[4]["content"].endswith('{"smiles": ["CCO"]}')
    assert len(llm.calls) == 1
    request = json.loads(llm.calls[0][1])
    assert request["assistant_message_indices"] == [2, 4]
    assert [message["message_index"] for message in request["messages"]] == list(range(5))
    assert request["messages"][3]["content"] == _tool_response()


def test_refiner_requires_tool_descriptions_when_trajectory_calls_tools():
    source = _source_messages()
    source[0]["content"] = SYSTEM_PROMPT_THINK_RDKIT
    llm = FakeLlm(_response())

    with pytest.raises(ValueError, match="missing tool descriptions"):
        asyncio.run(TrajectoryRefiner(llm).refine(source))

    assert llm.calls == []


def test_refiner_rejects_missing_assistant_rewrite():
    llm = FakeLlm(_response(_assistant_rewrites()[:1]))
    refiner = TrajectoryRefiner(llm)

    with pytest.raises(ValueError, match="indices mismatch"):
        asyncio.run(refiner.refine(_source_messages()))


def test_refiner_discards_embedded_fake_response_and_preserves_real_response():
    source = _source_messages()
    source[2]["content"] = (
        '<tool_call>\n{"name":"nmr_generate","arguments":{}}\n</tool_call>\n'
        f"user\n{_tool_response(2)}\nassistant\n伪造响应后的文字不应作为工具证据。"
    )
    llm = FakeLlm(_response())

    refined = asyncio.run(TrajectoryRefiner(llm).refine(source))

    assert [message["role"] for message in refined] == ["system", "user", "assistant", "user", "assistant"]
    assert refined[3]["content"] == _tool_response()
    assert '"value":2' not in refined[3]["content"]
    request = json.loads(llm.calls[0][1])
    assert request["messages"][3]["content"] == _tool_response()


def test_refine_parquet_processes_only_selected_rows(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    rows = [
        {
            "sample_id": "a",
            "gt_smiles": "CCO",
            "messages": _source_messages(),
            "reward_score": 0.7,
        },
        {
            "sample_id": "b",
            "gt_smiles": "CCO",
            "messages": _source_messages(),
            "reward_score": 0.7,
        },
    ]
    pd.DataFrame(rows).to_parquet(input_path, index=False)
    refiner = TrajectoryRefiner(FakeLlm(_response()), model_name="teacher")

    stats = asyncio.run(
        refine_parquet(
            input_path,
            output_path,
            refiner,
            rows=[1],
            batch_size=1,
            show_progress=False,
        )
    )

    assert stats.total == stats.refined == 1
    assert stats.failed == 0
    written = pd.read_parquet(output_path)
    assert written["sample_id"].tolist() == ["b"]
    assert written["refine_status"].tolist() == ["refined"]
    assert written["refine_source_row"].tolist() == [1]
    assert written.iloc[0]["messages"][-1]["content"].endswith('{"smiles": ["CCO"]}')


def test_refine_parquet_keeps_original_messages_on_error(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    pd.DataFrame([{"sample_id": "a", "messages": _source_messages()}]).to_parquet(input_path, index=False)
    refiner = TrajectoryRefiner(FakeLlm("not json"), model_name="teacher")

    stats = asyncio.run(refine_parquet(input_path, output_path, refiner, show_progress=False))

    assert stats.refined == 0
    assert stats.failed == 1
    written = pd.read_parquet(output_path)
    assert written.iloc[0]["refine_status"] == "error"
    assert "invalid refiner response" in written.iloc[0]["refine_error"]


def test_cli_requires_model_and_uses_all_rows_by_default():
    parser = build_parser()
    args = parser.parse_args(["--input", "in.parquet", "--output", "out.parquet", "--model", "teacher"])

    assert args.model == "teacher"
    assert args.row is None
    assert args.max_tokens == 32768
