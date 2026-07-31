"""Compile Spectune datasets into versioned training artifacts for external trainers.

Artifacts are a **stable handoff format**, not a copy of any trainer's internal
config. The verl adapter consumes these rows (Parquet / JSONL) and maps them
onto ``RLHFDataset`` fields; other backends can do the same.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from spectune.format.v1 import FORMAT_SPEC_VERSION, SYSTEM_PROMPT

JsonDict = dict[str, Any]

DEFAULT_DATA_SOURCE = "spectune/nmrexp"
DEFAULT_ABILITY = "structure_elucidation"
DEFAULT_AGENT_NAME = "tool_agent"


@dataclass(frozen=True, slots=True)
class ArtifactCompileConfig:
    """Options for mapping Spectune JSONL samples to training artifact rows."""

    data_source: str = DEFAULT_DATA_SOURCE
    ability: str = DEFAULT_ABILITY
    agent_name: str = DEFAULT_AGENT_NAME
    system_prompt: str = SYSTEM_PROMPT
    format_spec: str = FORMAT_SPEC_VERSION
    tool_names: tuple[str, ...] = ()
    reward_config: Mapping[str, Any] = field(default_factory=dict)
    # Schemas are large and identical across rows. Prefer resolving them at
    # reward time from ``tool_names`` via ``ToolManager``. Set True only when
    # the training host cannot import Spectune tools.
    include_tool_schemas: bool = False


def compile_sample(
    sample: Mapping[str, Any],
    *,
    index: int = 0,
    split: str = "train",
    config: ArtifactCompileConfig | None = None,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Convert one Spectune augmentor sample into a training artifact row."""
    config = config or ArtifactCompileConfig()
    turns = sample.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, str | bytes):
        raise ValueError("sample must contain a turns list")

    prompt: list[JsonDict] = [{"role": "system", "content": config.system_prompt}]
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError("each turn must be a mapping with role/content")
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("each turn requires string role and content")
        prompt.append({"role": role, "content": content})

    gt_smiles = sample.get("gt_smiles")
    if not isinstance(gt_smiles, str) or not gt_smiles.strip():
        raise ValueError("sample must contain a non-empty gt_smiles string")

    tool_names = tuple(config.tool_names)
    # PyArrow cannot encode empty structs (``{}``) to Parquet. Spectune tools
    # currently need no per-sample create state, so keep a null create_kwargs
    # placeholder that verl's ``.get("create_kwargs", {})`` / SpectuneTool both
    # treat as an empty mapping.
    tools_kwargs = {name: {"create_kwargs": None} for name in tool_names} if tool_names else None
    extra_info: JsonDict = {
        "split": split,
        "index": index,
        "sample_id": sample.get("sample_id"),
        "source_sample_id": sample.get("source_sample_id"),
        "format_spec": config.format_spec,
        "tool_names": list(tool_names),
        "need_tools_kwargs": bool(tool_names),
        "tools_kwargs": tools_kwargs,
    }
    if config.reward_config:
        extra_info["reward_config"] = dict(config.reward_config)

    if config.include_tool_schemas:
        resolved_schemas = tool_schemas
        if resolved_schemas is None and tool_names:
            from spectune.tools.catalog import schemas_for_names

            resolved_schemas = schemas_for_names(tool_names)
        if resolved_schemas is not None:
            extra_info["tool_schemas"] = [dict(schema) for schema in resolved_schemas]

    return {
        "data_source": config.data_source,
        "agent_name": config.agent_name,
        "prompt": prompt,
        "ability": config.ability,
        "reward_model": {"style": "rule", "ground_truth": gt_smiles.strip()},
        "extra_info": extra_info,
    }


def iter_compiled_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    split: str = "train",
    config: ArtifactCompileConfig | None = None,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> Iterable[JsonDict]:
    """Yield compiled rows for an iterable of Spectune samples."""
    config = config or ArtifactCompileConfig()
    for index, sample in enumerate(samples):
        yield compile_sample(
            sample,
            index=index,
            split=split,
            config=config,
            tool_schemas=tool_schemas,
        )


def load_jsonl(path: str | Path) -> list[JsonDict]:
    """Load a Spectune JSONL dataset."""
    records: list[JsonDict] = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write compiled artifacts as JSONL (always available, no extra deps)."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False))
            handle.write("\n")


def write_parquet(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write compiled artifacts as parquet for verl ``RLHFDataset``.

    Requires ``pip install spectune[data]`` (pandas + pyarrow).
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "writing parquet requires pandas/pyarrow; install with `pip install spectune[data]`"
        ) from exc

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(record) for record in records]).to_parquet(destination, index=False)


def compile_jsonl_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    split: str = "train",
    config: ArtifactCompileConfig | None = None,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    fmt: str | None = None,
) -> int:
    """Compile a Spectune JSONL file to JSONL or parquet artifacts.

    ``fmt`` defaults to the output suffix (``.parquet`` or ``.jsonl``).
    """
    config = config or ArtifactCompileConfig()
    samples = load_jsonl(input_path)
    records = list(
        iter_compiled_samples(
            samples,
            split=split,
            config=config,
            tool_schemas=tool_schemas,
        )
    )
    destination = Path(output_path)
    resolved_fmt = (fmt or destination.suffix.lstrip(".") or "jsonl").lower()
    if resolved_fmt in {"parquet", "pq"}:
        write_parquet(records, destination)
    elif resolved_fmt in {"jsonl", "json"}:
        write_jsonl(records, destination)
    else:
        raise ValueError(f"unsupported artifact format: {resolved_fmt!r}")
    return len(records)


def config_to_dict(config: ArtifactCompileConfig) -> JsonDict:
    """Serialize compile config for manifests / debugging."""
    payload = asdict(config)
    payload["reward_config"] = dict(config.reward_config)
    payload["tool_names"] = list(config.tool_names)
    return payload


__all__ = [
    "DEFAULT_ABILITY",
    "DEFAULT_AGENT_NAME",
    "DEFAULT_DATA_SOURCE",
    "ArtifactCompileConfig",
    "compile_jsonl_file",
    "compile_sample",
    "config_to_dict",
    "iter_compiled_samples",
    "load_jsonl",
    "write_jsonl",
    "write_parquet",
]
