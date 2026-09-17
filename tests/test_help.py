import asyncio

from spectune.tools.base import Tool
from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES
from spectune.tools.code_interpreter import (
    CODE_INTERPRETER_GUIDE,
    CodeInterpreterGuideTool,
    CodeInterpreterTool,
)
from spectune.tools.config import ToolManagerConfig
from spectune.tools.help import HelpTool
from spectune.tools.manager import ToolManager
from spectune.tools.web_search import WEB_SEARCH_GUIDE, WebSearchGuideTool, WebSearchTool


class _DummyGuideTool(HelpTool):
    name = "read_dummy_guide"
    description = "Dummy guide for testing HelpTool."
    guide = "dummy guide text"


def test_help_tool_execute_returns_guide_without_side_effects():
    tool = _DummyGuideTool()

    result = asyncio.run(tool.execute({}))

    assert result.completion == "success"
    assert result.status == "ok"
    assert result.data == {"guide": "dummy guide text"}


def test_help_tool_parameters_take_no_arguments():
    tool = _DummyGuideTool()

    assert tool.parameters == {"type": "object", "properties": {}, "additionalProperties": False}
    assert tool.schema["function"]["parameters"].get("required", []) == []
    assert tool.schema["function"]["parameters"]["properties"] == {}


def test_web_search_guide_tool_is_a_help_tool_not_a_plain_action_tool():
    guide_tool = WebSearchGuideTool()
    search_tool = WebSearchTool()

    assert isinstance(guide_tool, HelpTool)
    assert not isinstance(search_tool, HelpTool)
    assert isinstance(guide_tool, Tool) and isinstance(search_tool, Tool)


def test_web_search_guide_tool_returns_the_full_guide_text():
    tool = WebSearchGuideTool()

    result = asyncio.run(tool.execute({}))

    assert result.data["guide"] == WEB_SEARCH_GUIDE
    assert "NMR" in WEB_SEARCH_GUIDE
    assert "SMILES" in WEB_SEARCH_GUIDE


def test_tool_manager_from_config_registers_the_guide_tool_alongside_web_search():
    manager = ToolManager.from_config(ToolManagerConfig())

    assert "web_search" in manager.names
    assert "read_web_search_guide" in manager.names
    assert isinstance(manager.get("read_web_search_guide"), HelpTool)


def test_default_rl_tool_names_includes_the_guide_tool():
    assert "web_search" in DEFAULT_RL_TOOL_NAMES
    assert "read_web_search_guide" in DEFAULT_RL_TOOL_NAMES


def test_code_interpreter_guide_tool_is_a_help_tool_not_a_plain_action_tool():
    guide_tool = CodeInterpreterGuideTool()
    action_tool = CodeInterpreterTool()

    assert isinstance(guide_tool, HelpTool)
    assert not isinstance(action_tool, HelpTool)
    assert isinstance(guide_tool, Tool) and isinstance(action_tool, Tool)


def test_code_interpreter_guide_tool_returns_the_full_guide_text():
    tool = CodeInterpreterGuideTool()

    result = asyncio.run(tool.execute({}))

    assert result.completion == "success"
    assert result.status == "ok"
    assert result.data["guide"] == CODE_INTERPRETER_GUIDE
    assert "CalcMolFormula" in CODE_INTERPRETER_GUIDE
    assert "Fragments" in CODE_INTERPRETER_GUIDE


def test_tool_manager_from_config_registers_the_code_interpreter_guide_tool():
    manager = ToolManager.from_config(ToolManagerConfig())

    assert "code_interpreter" in manager.names
    assert "read_code_interpreter_guide" in manager.names
    assert isinstance(manager.get("read_code_interpreter_guide"), HelpTool)


def test_default_rl_tool_names_includes_the_code_interpreter_guide_tool():
    assert "code_interpreter" in DEFAULT_RL_TOOL_NAMES
    assert "read_code_interpreter_guide" in DEFAULT_RL_TOOL_NAMES
