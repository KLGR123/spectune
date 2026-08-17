import asyncio

import pytest

from spectune.tools import ToolManager
from spectune.tools.base import Tool, ToolResult
from spectune.tools.config import NMR_GENERATE_MAX_TOPK, NmrGenerateConfig, ToolManagerConfig


class _EchoTool(Tool):
    name = "echo"
    description = "Echo arguments back."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    async def execute(self, arguments):
        self.calls.append(dict(arguments))
        if self.fail:
            return ToolResult(completion="failure", status="error", warnings=["boom"])
        return ToolResult(completion="success", status="ok", data={"echo": dict(arguments)})


class _BoomTool(_EchoTool):
    async def execute(self, arguments):
        raise RuntimeError("kaboom")


class TestToolManagerInvoke:
    def test_unknown_tool_returns_error_without_raising(self):
        manager = ToolManager([_EchoTool()])

        result = asyncio.run(manager.invoke("does_not_exist", {}))

        assert result.completion == "failure"
        assert result.status == "error"
        assert result.warnings == ["unknown tool: does_not_exist"]

    def test_invalid_json_string_arguments_returns_error(self):
        manager = ToolManager([_EchoTool()])

        result = asyncio.run(manager.invoke("echo", "{not json"))

        assert result.completion == "failure"
        assert "invalid JSON arguments" in result.warnings[0]

    def test_non_dict_json_arguments_returns_error(self):
        manager = ToolManager([_EchoTool()])

        result = asyncio.run(manager.invoke("echo", "[1, 2, 3]"))

        assert result.completion == "failure"
        assert result.warnings == ["tool arguments must be a JSON object"]

    def test_json_string_arguments_are_parsed_before_execute(self):
        tool = _EchoTool()
        manager = ToolManager([tool])

        result = asyncio.run(manager.invoke("echo", '{"x": 1}'))

        assert result.completion == "success"
        assert tool.calls == [{"x": 1}]

    def test_none_arguments_are_passed_as_empty_dict(self):
        tool = _EchoTool()
        manager = ToolManager([tool])

        result = asyncio.run(manager.invoke("echo", None))

        assert result.completion == "success"
        assert tool.calls == [{}]

    def test_tool_exception_is_captured_as_a_failure_result(self):
        manager = ToolManager([_BoomTool()])

        result = asyncio.run(manager.invoke("echo", {}))

        assert result.completion == "failure"
        assert "RuntimeError: kaboom" in result.warnings[0]

    def test_prints_warning_only_once_threshold_is_reached(self, capsys):
        manager = ToolManager([_EchoTool(fail=True)])
        manager.FAILURE_WARN_THRESHOLD = 2

        asyncio.run(manager.invoke("echo", {}))
        assert capsys.readouterr().out == ""

        asyncio.run(manager.invoke("echo", {}))
        assert "failed 2 consecutive times" in capsys.readouterr().out

    def test_prints_recovery_message_after_a_success_following_failures(self, capsys):
        tool = _EchoTool(fail=True)
        manager = ToolManager([tool])

        asyncio.run(manager.invoke("echo", {}))
        tool.fail = False
        asyncio.run(manager.invoke("echo", {}))

        assert "recovered after 1 failures" in capsys.readouterr().out


class TestNmrGenTopkValidation:
    @pytest.mark.parametrize("topk", [0, NMR_GENERATE_MAX_TOPK + 1])
    def test_nmr_generate_config_rejects_out_of_range_default_topk(self, topk):
        with pytest.raises(ValueError, match="default_topk"):
            NmrGenerateConfig(default_topk=topk)

    @pytest.mark.parametrize("topk", [0, NMR_GENERATE_MAX_TOPK + 1])
    def test_tool_manager_config_rejects_out_of_range_nmr_gen_topk(self, topk):
        with pytest.raises(ValueError, match="nmr_gen_topk"):
            ToolManagerConfig(nmr_gen_topk=topk)

    def test_tool_manager_config_overrides_nmr_generate_default_topk(self):
        config = ToolManagerConfig(nmr_gen_topk=5)

        assert config.nmr_generate.default_topk == 5
