"""Canonical JSON-Lines read/write helpers.

Every pipeline stage that persists ``list[dict]`` records as `.jsonl` --
dataloader truth pools, augmentor output, rollout dumps, compiled artifacts --
used to carry its own copy of this loop, each with slightly different
validation (some checked line numbers on malformed JSON, some required a
JSON object, some silently accepted anything ``json.loads`` returned). This
module is the one place that logic lives; call sites either use it directly
or wrap it for a more specific return type (e.g. domain objects instead of
plain dicts).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


def read_jsonl(path: str | Path, *, require_mapping: bool = True) -> list[JsonDict]:
    """Fully materialize a JSON-Lines file into a list of dicts.

    Raises ``ValueError`` naming the offending line number on malformed JSON,
    or (when ``require_mapping``, the default) on a line whose JSON value
    isn't an object.
    """
    records: list[JsonDict] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if require_mapping and not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write ``records`` to ``path`` as JSON-Lines, returning the row count written."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


__all__ = ["read_jsonl", "write_jsonl"]
