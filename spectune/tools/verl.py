"""Optional verl adapter for Spectune tools.

Spectune tools stay framework-agnostic. This module is the plugin surface that
verl loads from ``tool_config_path`` YAML::

    tools:
      - class_name: spectune.tools.verl.SpectuneTool
        config:
          type: native
          tool_name: nmr_rerank

Hard ``verl`` imports happen only inside methods that run under a verl worker,
so installing Spectune without verl remains valid.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from spectune.tools.base import compact_tool_payload
from spectune.tools.catalog import (
    DEFAULT_RL_TOOL_NAMES,
    openai_schema_for,
    reset_shared_manager,
    resolve_tool_names,
    schemas_for_names,
    shared_manager,
)
from spectune.tools.config import NMR_GENERATE_MAX_TOPK, ToolManagerConfig
from spectune.tools.manager import ToolManager

JsonDict = dict[str, Any]


def build_tools_config(
    tool_names: Sequence[str] | None = None,
    *,
    manager: Any = None,
    class_name: str = "spectune.tools.verl.SpectuneTool",
    nmr_gen_topk: int | None = None,
) -> JsonDict:
    """Build a verl ``tools:`` config mapping (ready to dump as YAML).

    Entries are schema-free: ``SpectuneTool`` loads the full JSON Schema tree
    from :class:`~spectune.tools.ToolManager` at init. Do **not** embed schemas
    in YAML — verl's pydantic models silently drop ``items`` / ``minimum`` / …

    ``nmr_gen_topk`` (1..NMR_GENERATE_MAX_TOPK) is recorded on the
    ``nmr_generate`` entry; ``SpectuneTool`` applies it as the tool's default
    ``topk`` at init.
    """
    if nmr_gen_topk is not None and not 1 <= nmr_gen_topk <= NMR_GENERATE_MAX_TOPK:
        raise ValueError(f"nmr_gen_topk must be in [1, {NMR_GENERATE_MAX_TOPK}]")
    names = resolve_tool_names(tool_names, manager=manager)
    entries = []
    for name in names:
        config = {"type": "native", "tool_name": name}
        if nmr_gen_topk is not None and name == "nmr_generate":
            config["nmr_gen_topk"] = nmr_gen_topk
        entries.append({"class_name": class_name, "config": config})
    return {"tools": entries}


def _yaml_escape(value: str) -> str:
    if value == "":
        return '""'
    special = any(ch in value for ch in ":#{}[]&*!|>'\"%@`,\n") or value.strip() != value
    if special:
        return json.dumps(value, ensure_ascii=False)
    return value


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return _yaml_escape(str(value))


def _dump_yaml(value: Any, *, indent: int = 0) -> list[str]:
    """Minimal YAML emitter for the nested dict/list shapes we generate."""
    prefix = " " * indent
    if isinstance(value, Mapping):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, child in value.items():
            key_text = str(key)
            if isinstance(child, Mapping | list):
                lines.append(f"{prefix}{key_text}:")
                lines.extend(_dump_yaml(child, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key_text}: {_yaml_scalar(child)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, Mapping | list):
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def tools_config_to_yaml(config: Mapping[str, Any]) -> str:
    """Serialize a tools config mapping to YAML text."""
    return "\n".join(_dump_yaml(dict(config))) + "\n"


def write_tools_config(
    path: str | Path,
    tool_names: Sequence[str] | None = None,
    *,
    manager: Any = None,
    nmr_gen_topk: int | None = None,
) -> Path:
    """Write a verl ``tool_config_path`` YAML for Spectune tools."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        tools_config_to_yaml(build_tools_config(tool_names, manager=manager, nmr_gen_topk=nmr_gen_topk)),
        encoding="utf-8",
    )
    return destination


class PreservedOpenAIToolSchema:
    """OpenAI tool schema carrier that keeps full JSON Schema parameter trees.

    verl's ``OpenAIFunctionToolSchema`` pydantic models only keep
    ``type``/``description``/``enum`` on properties and silently drop
    ``items``, ``minimum``, ``additionalProperties``, etc. Chat templates and
    reward judges both need the intact Spectune schemas.
    """

    def __init__(self, schema: Mapping[str, Any]) -> None:
        if not isinstance(schema, Mapping):
            raise TypeError("schema must be a mapping")
        raw = copy.deepcopy(dict(schema))
        function = raw.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("schema.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("schema.function.name is required")
        self.type = str(raw.get("type") or "function")
        self.function = _FunctionView(
            name=name.strip(),
            description=str(function.get("description") or ""),
            parameters=copy.deepcopy(dict(function.get("parameters") or {"type": "object", "properties": {}})),
            strict=bool(function.get("strict", False)),
        )
        self._raw = {
            "type": self.type,
            "function": {
                "name": self.function.name,
                "description": self.function.description,
                "parameters": self.function.parameters,
                "strict": self.function.strict,
            },
        }

    def model_dump(self, *args: Any, **kwargs: Any) -> JsonDict:
        # Ignore exclude_unset / exclude_none — those flags exist for pydantic
        # models; dropping None here would strip legitimate schema defaults.
        del args, kwargs
        return copy.deepcopy(self._raw)


class _FunctionView:
    __slots__ = ("name", "description", "parameters", "strict")

    def __init__(self, *, name: str, description: str, parameters: JsonDict, strict: bool) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.strict = strict


class SpectuneTool:
    """Duck-typed verl ``BaseTool`` wrapper around :class:`~spectune.tools.ToolManager`.

    Instantiated by verl from YAML. Schemas always come from ``ToolManager``
    (full JSON Schema). Any verl-prevalidated ``tool_schema`` argument is
    ignored because those pydantic models strip parameter constraints.
    """

    def __init__(self, config: dict, tool_schema: Any = None) -> None:
        del tool_schema  # Intentionally ignored; see class docstring.
        self.config = config or {}
        tool_name = self.config.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("SpectuneTool config.tool_name is required")
        self._tool_name = tool_name.strip()
        # A run-level nmr_gen_topk (from tools_config.yaml) overrides the
        # nmr_generate default topk.  Use a dedicated manager so both the
        # schema default and execution honor it without touching the shared one.
        topk = self.config.get("nmr_gen_topk")
        self._manager = None
        if topk is not None:
            self._manager = ToolManager.from_config(ToolManagerConfig(nmr_gen_topk=topk))
        manager = self._manager or shared_manager()
        self.tool_schema = PreservedOpenAIToolSchema(openai_schema_for(self._tool_name, manager=manager))
        self.name = self.tool_schema.function.name
        # Per-trajectory state keyed by instance_id (reserved for create_kwargs).
        self._instances: dict[str, dict[str, Any]] = {}

    def get_openai_tool_schema(self) -> Any:
        return self.tool_schema

    async def create(
        self,
        instance_id: str | None = None,
        create_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[str, Any]:
        """Create a trajectory-local tool instance.

        verl's ``ToolAgentLoop`` calls ``create(create_kwargs=...)``. Accept both
        that form and the ``BaseTool`` ``(instance_id, **kwargs)`` form.
        """
        from verl.tools.schemas import ToolResponse

        if create_kwargs is None:
            nested = kwargs.get("create_kwargs")
            create_kwargs = nested if isinstance(nested, Mapping) else {}
        # Parquet may round-trip unused create_kwargs as null.
        state = dict(create_kwargs) if isinstance(create_kwargs, Mapping) else {}
        resolved_id = instance_id or str(uuid4())
        self._instances[resolved_id] = state
        return resolved_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: Any) -> tuple[Any, float, dict]:
        from verl.tools.schemas import ToolResponse

        del kwargs
        try:
            manager = self._manager or shared_manager()
            result = await manager.invoke(self._tool_name, parameters or {})
            text = json.dumps(compact_tool_payload(result), ensure_ascii=False)
            metrics = {
                "completion": result.completion,
                "status": result.status,
                "tool_name": self._tool_name,
            }
            return ToolResponse(text=text), 0.0, metrics
        except Exception as exc:  # noqa: BLE001 - surface tool failures to the model, not the worker
            text = json.dumps(
                {
                    "completion": "failure",
                    "status": "error",
                    "warnings": [f"{type(exc).__name__}: {exc}"],
                },
                ensure_ascii=False,
            )
            return (
                ToolResponse(text=text),
                0.0,
                {"completion": "failure", "status": "error", "tool_name": self._tool_name},
            )

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        del instance_id, kwargs
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        del kwargs
        self._instances.pop(instance_id, None)


__all__ = [
    "DEFAULT_RL_TOOL_NAMES",
    "PreservedOpenAIToolSchema",
    "SpectuneTool",
    "build_tools_config",
    "openai_schema_for",
    "reset_shared_manager",
    "resolve_tool_names",
    "schemas_for_names",
    "tools_config_to_yaml",
    "write_tools_config",
]
