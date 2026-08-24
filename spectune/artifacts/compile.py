"""Compile Spectune datasets into versioned training artifacts for external trainers.

Artifacts are a **stable handoff format**, not a copy of any trainer's internal
config. The verl adapter consumes these rows (Parquet / JSONL) and maps them
onto ``RLHFDataset`` fields; other backends can do the same.

`raw_prompt` is for verl tool_agent.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from spectune.augmentor.config import INFORMATION_TYPES
from spectune.format.v1 import FORMAT_SPEC_VERSION, SYSTEM_PROMPT
from spectune.jsonl import read_jsonl as _read_jsonl
from spectune.jsonl import write_jsonl as _write_jsonl

JsonDict = dict[str, Any]

DEFAULT_DATA_SOURCE = "spectune/nmrexp"
DEFAULT_ABILITY = "structure_elucidation"
DEFAULT_AGENT_NAME = "tool_agent"

_CANONICAL_INFORMATION_TYPES = frozenset(INFORMATION_TYPES)


def infer_data_type(sample_id: str, augmentation: Any) -> str:
    """Resolve a canonical information-type label (one of ``INFORMATION_TYPES``).

    Written once at compile time into ``extra_info.data_type`` so downstream
    consumers (eval, debug viewers) never need to re-derive it.

    NMRexp samples (from ``spectune.augmentor``) carry the type in
    ``augmentation["information_type"]``. Samples from other sources (e.g.
    the "others" SFT pool, which has no ``augmentation`` dict) encode it as
    the first colon-delimited segment of ``sample_id``, which must be an
    exact match against ``INFORMATION_TYPES`` -- unresolved compound prefixes
    (e.g. the former "none_plus_formula", which mixed rows that stated a
    molecular formula with rows that didn't) should be split and relabeled
    at the source rather than guessed here.
    """
    if isinstance(augmentation, Mapping):
        candidate = augmentation.get("information_type")
        if isinstance(candidate, str) and candidate in _CANONICAL_INFORMATION_TYPES:
            return candidate
    prefix = str(sample_id).split(":", 1)[0]
    if prefix in _CANONICAL_INFORMATION_TYPES:
        return prefix
    return "none"


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

    user_turns: list[JsonDict] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError("each turn must be a mapping with role/content")
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("each turn requires string role and content")
        if role == "user":
            user_turns.append({"role": role, "content": content})

    if not user_turns:
        raise ValueError("sample must contain at least one user turn")

    # raw_prompt only carries the first user turn so verl's tool_agent_loop
    # starts generation from [system, user1].  Any subsequent user turns are
    # delivered interactively via ScriptedFollowupInteraction.
    prompt: list[JsonDict] = [{"role": "system", "content": config.system_prompt}] + user_turns
    raw_prompt: list[JsonDict] = [{"role": "system", "content": config.system_prompt}, user_turns[0]]

    gt_smiles = sample.get("gt_smiles")
    if not isinstance(gt_smiles, str) or not gt_smiles.strip():
        raise ValueError("sample must contain a non-empty gt_smiles string")

    tool_names = tuple(config.tool_names)
    # PyArrow cannot encode empty structs (``{}``) to Parquet. Spectune tools
    # currently need no per-sample create state, so keep a null create_kwargs
    # placeholder that verl's ``.get("create_kwargs", {})`` / SpectuneTool both
    # treat as an empty mapping.
    tools_kwargs = {name: {"create_kwargs": None} for name in tool_names} if tool_names else None
    # interaction_kwargs is always present so that ToolAgentLoop can activate
    # ScriptedFollowupInteraction for every sample.  followup_turns is empty
    # for single-turn samples, causing the interaction to terminate immediately.
    interaction_kwargs: JsonDict = {
        "name": "scripted_followup",
        "followup_turns": [t["content"] for t in user_turns[1:]],
    }
    augmentation = sample.get("augmentation")
    sample_id = sample.get("sample_id")

    extra_info: JsonDict = {
        "split": split,
        "index": index,
        "sample_id": sample_id,
        # Canonical scenario label, resolved once here so every consumer
        # (eval, debug viewers) reads the same field instead of each
        # re-deriving it from sample_id conventions that differ by source.
        # This is the *only* part of the source's scenario metadata (e.g.
        # the augmentor's `augmentation` dict) that survives compilation --
        # everything else has no downstream reader, so it is not carried
        # into extra_info at all.
        "data_type": infer_data_type(str(sample_id or ""), augmentation),
        # Full multi-turn user conversation, independent of verl's
        # ``prompt``/``raw_prompt`` split (which intentionally excludes
        # follow-up turns from ``raw_prompt`` — see ScriptedFollowupInteraction).
        # Offline consumers (spectune.rollout, spectune.rollout.eval) read
        # this instead of guessing which verl column holds the full
        # conversation.
        "turns": user_turns,
        "format_spec": config.format_spec,
        "tool_names": list(tool_names),
        "need_tools_kwargs": bool(tool_names),
        "tools_kwargs": tools_kwargs,
        "interaction_kwargs": interaction_kwargs,
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
        # Per-file override from load_jsonl_files(data_sources=...), else the
        # config-wide default -- lets one compile run tag NMRexp and "others"
        # rows differently without a separate invocation per source file.
        "data_source": sample.get("_data_source") or config.data_source,
        "agent_name": config.agent_name,
        "prompt": prompt,
        "raw_prompt": raw_prompt,
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
    return _read_jsonl(path)


def write_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write compiled artifacts as JSONL (always available, no extra deps)."""
    _write_jsonl(path, records)


def write_parquet(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write compiled artifacts as parquet for verl ``RLHFDataset``.

    Requires ``pip install spectune[data]`` (pandas + pyarrow).
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError("writing parquet requires pandas/pyarrow; install with `pip install spectune[data]`") from exc

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(record) for record in records]).to_parquet(destination, index=False)


def load_jsonl_files(
    paths: str | Path | Iterable[str | Path],
    *,
    shuffle: bool = True,
    seed: int | None = None,
    data_sources: str | Sequence[str] | None = None,
) -> list[JsonDict]:
    """Load one or more Spectune JSONL files and merge them into one list.

    When multiple ``paths`` are given, records from all files are concatenated
    and then shuffled together (``shuffle=True`` by default) so downstream
    consumers see an interleaved mix rather than each source's rows in a
    contiguous block. Pass ``seed`` for a reproducible shuffle order.

    ``data_sources``, when given, tags every record loaded from a given path
    with that path's own ``data_source`` (stashed under the private
    ``"_data_source"`` key, read by :func:`compile_sample` to override
    ``config.data_source`` just for that row) -- e.g. compile the NMRexp
    augmentor output and the "others" SFT pool in one run while still telling
    them apart downstream. Pass one value (broadcast to every path) or
    exactly one value per path; omit to leave every row on ``config.data_source``.
    """
    if isinstance(paths, str | Path):
        path_list: list[str | Path] = [paths]
    else:
        path_list = list(paths)
    if not path_list:
        raise ValueError("at least one input path is required")

    if data_sources is None:
        source_list: list[str | None] = [None] * len(path_list)
    elif isinstance(data_sources, str):
        source_list = [data_sources] * len(path_list)
    else:
        source_list = list(data_sources)
        if len(source_list) == 1:
            source_list = source_list * len(path_list)
        elif len(source_list) != len(path_list):
            raise ValueError(
                f"got {len(source_list)} data_sources for {len(path_list)} input paths; "
                "pass exactly one (broadcast to all) or one per path"
            )

    records: list[JsonDict] = []
    for path, source in zip(path_list, source_list, strict=True):
        loaded = load_jsonl(path)
        if source is not None:
            for record in loaded:
                record["_data_source"] = source
        records.extend(loaded)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    return records


def compile_jsonl_file(
    input_path: str | Path | Iterable[str | Path],
    output_path: str | Path,
    *,
    split: str = "train",
    config: ArtifactCompileConfig | None = None,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    fmt: str | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    data_sources: str | Sequence[str] | None = None,
) -> int:
    """Compile a Spectune JSONL file (or files) to JSONL or parquet artifacts.

    ``input_path`` accepts a single path or an iterable of paths. When
    multiple paths are given, their samples are merged and shuffled together
    (see ``load_jsonl_files``) before being compiled into one output file.
    ``data_sources`` optionally tags each input path's rows with its own
    ``data_source`` value instead of ``config.data_source`` uniformly (see
    ``load_jsonl_files``).

    ``fmt`` defaults to the output suffix (``.parquet`` or ``.jsonl``).
    """
    config = config or ArtifactCompileConfig()
    samples = load_jsonl_files(input_path, shuffle=shuffle, seed=seed, data_sources=data_sources)
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


_INTERACTION_CONFIG_TEMPLATE = (
    "interaction:\n  - class_name: spectune.artifacts.followup.ScriptedFollowupInteraction\n    name: scripted_followup\n    config: {}\n"
)


def write_interaction_config(path: str | Path) -> None:
    """Write the verl interaction config YAML for ScriptedFollowupInteraction."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_INTERACTION_CONFIG_TEMPLATE, encoding="utf-8")


__all__ = [
    "DEFAULT_ABILITY",
    "DEFAULT_AGENT_NAME",
    "DEFAULT_DATA_SOURCE",
    "ArtifactCompileConfig",
    "compile_jsonl_file",
    "compile_sample",
    "config_to_dict",
    "infer_data_type",
    "iter_compiled_samples",
    "load_jsonl",
    "load_jsonl_files",
    "write_interaction_config",
    "write_jsonl",
    "write_parquet",
]
